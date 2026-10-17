package com.sih26168.deadreckoning

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.CacheControl
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.nio.FloatBuffer
import java.util.concurrent.TimeUnit

/**
 * Model-only updates: lets a newer BiasCorrectionNet reach phones without
 * a new APK.
 *
 * On app start MainActivity calls checkForUpdate() in the background. It
 * fetches manifest.json from the public "models" bucket in the project's
 * Supabase Storage (see ModelManifest and scripts/publish_model.py) and,
 * if the version is newer than what would run now, downloads the model,
 * checks its size and SHA-256, opens it with ONNX Runtime, checks the
 * input/output shapes and runs one test window. Only then is it moved into
 * place. The new model is picked up the next time the app starts;
 * the session already running is never swapped mid-drive.
 *
 * BiasCorrectionModel asks downloadedModel() for a verified file at load
 * time and falls back to the model bundled in the APK on any failure, so
 * a bad download or a corrupted file can never stop the app from working.
 * The dead-reckoning pipeline itself stays fully offline; with no network
 * this whole class is a logged no-op.
 */
object ModelUpdater {

    /** Version of the model bundled in the APK's assets. Bump this
     * whenever checkpoints/dead_reckoning_model.onnx is replaced, so a
     * phone with the new APK doesn't keep an older downloaded model.
     * 1 = best.pt, 2 = best_v3chain.pt. */
    const val BUNDLED_MODEL_VERSION = 2

    private const val TAG = "ModelUpdater"
    private const val DIR_NAME = "model_update"
    private const val MODEL_FILE = "current.onnx"
    private const val PREFS_NAME = "model_update_prefs"
    private const val KEY_VERSION = "installed_version"
    private const val KEY_SHA256 = "installed_sha256"
    private const val KEY_REJECTED = "rejected_version"

    private val client = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .build()

    /** A downloaded model that passed every check, ready to load. */
    class Downloaded(val version: Int, val bytes: ByteArray)

    sealed class Result {
        object UpToDate : Result()
        data class Installed(val version: Int) : Result()
        data class Failed(val reason: String) : Result()
    }

    /**
     * The downloaded model, if there is one newer than the bundled model
     * and its bytes still match the SHA-256 recorded at install time.
     * Returns null (and cleans up) otherwise, so the caller loads the
     * bundled model.
     */
    fun downloadedModel(context: Context): Downloaded? {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val version = prefs.getInt(KEY_VERSION, 0)
        val sha = prefs.getString(KEY_SHA256, null)
        val file = File(File(context.filesDir, DIR_NAME), MODEL_FILE)
        if (version <= BUNDLED_MODEL_VERSION || sha == null || !file.exists()) {
            // Nothing installed, or the APK now bundles something as new.
            if (file.exists() || version != 0) clearInstalled(context, rejected = false)
            return null
        }
        return try {
            val bytes = file.readBytes()
            if (ModelChecks.sha256Hex(bytes) != sha) {
                // The file changed on disk after it was verified. Delete it
                // but don't blacklist the version: a fresh download is fine.
                Log.w(TAG, "downloaded model v$version failed its checksum on load, using bundled model")
                clearInstalled(context, rejected = false)
                null
            } else {
                Downloaded(version, bytes)
            }
        } catch (e: Exception) {
            Log.w(TAG, "could not read downloaded model v$version, using bundled model", e)
            clearInstalled(context, rejected = false)
            null
        }
    }

    /** Called when a downloaded model fails to load. Deletes it and
     * remembers its version so the same file isn't fetched again; a
     * later, higher version is still accepted. */
    fun discardDownloaded(context: Context) = clearInstalled(context, rejected = true)

    private fun clearInstalled(context: Context, rejected: Boolean) {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val version = prefs.getInt(KEY_VERSION, 0)
        File(File(context.filesDir, DIR_NAME), MODEL_FILE).delete()
        prefs.edit().apply {
            remove(KEY_VERSION)
            remove(KEY_SHA256)
            if (rejected && version > 0) putInt(KEY_REJECTED, version)
        }.commit()
    }

    /** Throws if this session's shapes aren't exactly what FusionEngine
     * feeds and reads (see ModelChecks). */
    fun requireCompatible(session: OrtSession) {
        val inputs = session.inputInfo.values.map { (it.info as TensorInfo).shape }
        val outputs = session.outputInfo.values.map { (it.info as TensorInfo).shape }
        require(ModelChecks.shapesCompatible(inputs, outputs)) {
            "model shapes ${inputs.map { it.toList() }} -> ${outputs.map { it.toList() }} " +
                "don't match ${ModelChecks.INPUT_SHAPE.toList()} -> ${ModelChecks.OUTPUT_SHAPE.toList()}"
        }
    }

    /** Never throws; every failure comes back as Result.Failed. */
    suspend fun checkForUpdate(context: Context): Result = withContext(Dispatchers.IO) {
        try {
            val result = check(context.applicationContext)
            when (result) {
                is Result.Installed -> Log.i(TAG, "installed model v${result.version}, active from next app start")
                is Result.UpToDate -> Log.i(TAG, "model up to date")
                is Result.Failed -> Log.w(TAG, "update rejected: ${result.reason}")
            }
            result
        } catch (e: Exception) {
            // Cancellation isn't a failed check: let it reach the caller.
            if (e is kotlin.coroutines.cancellation.CancellationException) throw e
            Log.w(TAG, "update check failed: $e")
            Result.Failed(e.toString())
        }
    }

    private fun check(context: Context): Result {
        val baseUrl = BuildConfig.MODEL_BASE_URL.trimEnd('/')
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

        val manifestJson = fetch("$baseUrl/manifest.json", noCache = true, maxBytes = 16 * 1024)
        val manifest = ModelManifest.parse(String(manifestJson, Charsets.UTF_8))

        val running = maxOf(BUNDLED_MODEL_VERSION, prefs.getInt(KEY_VERSION, 0))
        val rejected = prefs.getInt(KEY_REJECTED, 0).takeIf { it > 0 }
        if (!ModelChecks.shouldDownload(manifest.version, running, rejected)) return Result.UpToDate

        val bytes = fetch("$baseUrl/${manifest.file}", noCache = false, maxBytes = ModelManifest.MAX_MODEL_BYTES)
        if (bytes.size.toLong() != manifest.sizeBytes) {
            return Result.Failed("v${manifest.version}: got ${bytes.size} bytes, manifest says ${manifest.sizeBytes}")
        }
        val sha = ModelChecks.sha256Hex(bytes)
        if (sha != manifest.sha256) {
            return Result.Failed("v${manifest.version}: checksum mismatch")
        }
        try {
            validate(bytes)
        } catch (e: Exception) {
            // A model that downloads fine but can't run here would fail
            // the same way every time, so don't fetch this version again.
            prefs.edit().putInt(KEY_REJECTED, manifest.version).commit()
            return Result.Failed("v${manifest.version}: ${e.message}")
        }

        // Write to a temp file and rename, so a crash mid-write never
        // leaves a half-written current.onnx behind.
        val dir = File(context.filesDir, DIR_NAME).apply { mkdirs() }
        val tmp = File(dir, "$MODEL_FILE.tmp")
        tmp.writeBytes(bytes)
        val target = File(dir, MODEL_FILE)
        if (!tmp.renameTo(target)) {
            tmp.delete()
            return Result.Failed("v${manifest.version}: could not move model into place")
        }
        prefs.edit()
            .putInt(KEY_VERSION, manifest.version)
            .putString(KEY_SHA256, sha)
            .commit()
        return Result.Installed(manifest.version)
    }

    /** Opens the model, checks its shapes and runs one all-zero window,
     * which is a stationary phone. Throws on any problem. */
    private fun validate(bytes: ByteArray) {
        val env = OrtEnvironment.getEnvironment()
        env.createSession(bytes, OrtSession.SessionOptions()).use { session ->
            requireCompatible(session)
            val window = FloatArray(50 * 6)
            OnnxTensor.createTensor(env, FloatBuffer.wrap(window), ModelChecks.INPUT_SHAPE).use { input ->
                session.run(mapOf(session.inputNames.first() to input)).use { result ->
                    @Suppress("UNCHECKED_CAST")
                    val out = (result[0].value as Array<Array<FloatArray>>)[0]
                    require(out.size == 50 && out.all { it.size == 2 && it.all(Float::isFinite) }) {
                        "test inference gave a wrong-shaped or non-finite output"
                    }
                }
            }
        }
    }

    private fun fetch(url: String, noCache: Boolean, maxBytes: Long): ByteArray {
        val request = Request.Builder().url(url).apply {
            if (noCache) cacheControl(CacheControl.FORCE_NETWORK)
        }.build()
        client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) throw IllegalStateException("HTTP ${response.code} for $url")
            val body = response.body ?: throw IllegalStateException("empty body for $url")
            val length = body.contentLength()
            if (length > maxBytes) throw IllegalStateException("$url is $length bytes, limit $maxBytes")
            val bytes = body.byteStream().use { it.readNBytesCompat(maxBytes + 1) }
            if (bytes.size > maxBytes) throw IllegalStateException("$url is over $maxBytes bytes")
            return bytes
        }
    }

    private fun java.io.InputStream.readNBytesCompat(limit: Long): ByteArray {
        val out = java.io.ByteArrayOutputStream()
        val buf = ByteArray(64 * 1024)
        var total = 0L
        while (total < limit) {
            val n = read(buf, 0, minOf(buf.size.toLong(), limit - total).toInt())
            if (n < 0) break
            out.write(buf, 0, n)
            total += n
        }
        return out.toByteArray()
    }
}
