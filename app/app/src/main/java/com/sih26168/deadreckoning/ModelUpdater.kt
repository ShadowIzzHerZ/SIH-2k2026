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
 * Model-only updates: lets newer ensemble models reach phones without a
 * new APK.
 *
 * On app start MainActivity calls checkForUpdate() in the background. It
 * fetches manifest.json from the public "models" bucket in the project's
 * Supabase Storage (see ModelManifest, EnsembleManifest and
 * scripts/publish_model.py) and, for each ModelRole whose published
 * version is newer than what would run now, downloads that model, checks
 * its size and SHA-256, opens it with ONNX Runtime, checks the role's
 * input/output shapes and runs one test window. Only then is it moved
 * into place. The roles are independent: a bad download of one never
 * touches the other. A new model is picked up the next time the app
 * starts; the session already running is never swapped mid-drive.
 *
 * BiasCorrectionModel asks downloadedModel() for a verified file at load
 * time and falls back to the role's model bundled in the APK on any
 * failure, so a bad download or a corrupted file can never stop the app
 * from working. The dead-reckoning pipeline itself stays fully offline;
 * with no network this whole class is a logged no-op.
 */
object ModelUpdater {

    /** Version of each role's model bundled in the APK's assets. Bump one
     * whenever its checkpoints/model_<role>.onnx is replaced, so a phone
     * with the new APK doesn't keep an older downloaded model.
     * 1 = augbias_long (slow) and speedaware (fast), the first ensemble. */
    const val BUNDLED_SLOW_VERSION = 1
    const val BUNDLED_FAST_VERSION = 1

    fun bundledVersion(role: ModelRole): Int = when (role) {
        ModelRole.SLOW -> BUNDLED_SLOW_VERSION
        ModelRole.FAST -> BUNDLED_FAST_VERSION
    }

    private const val TAG = "ModelUpdater"
    private const val DIR_NAME = "model_update"
    private const val PREFS_NAME = "model_update_prefs"
    // The single model the app ran before the ensemble. Removed on the
    // first check so it doesn't sit on the phone forever.
    private const val LEGACY_MODEL_FILE = "current.onnx"
    private val LEGACY_KEYS = listOf("installed_version", "installed_sha256", "rejected_version")

    private fun modelFile(role: ModelRole) = "${role.key}.onnx"
    private fun keyVersion(role: ModelRole) = "${role.key}_installed_version"
    private fun keySha(role: ModelRole) = "${role.key}_installed_sha256"
    private fun keyRejected(role: ModelRole) = "${role.key}_rejected_version"

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
     * The downloaded model for [role], if there is one newer than the
     * bundled model and its bytes still match the SHA-256 recorded at
     * install time. Returns null (and cleans up) otherwise, so the caller
     * loads the bundled model.
     */
    fun downloadedModel(context: Context, role: ModelRole): Downloaded? {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val version = prefs.getInt(keyVersion(role), 0)
        val sha = prefs.getString(keySha(role), null)
        val file = File(File(context.filesDir, DIR_NAME), modelFile(role))
        if (version <= bundledVersion(role) || sha == null || !file.exists()) {
            // Nothing installed, or the APK now bundles something as new.
            if (file.exists() || version != 0) clearInstalled(context, role, rejected = false)
            return null
        }
        return try {
            val bytes = file.readBytes()
            if (ModelChecks.sha256Hex(bytes) != sha) {
                // The file changed on disk after it was verified. Delete it
                // but don't blacklist the version: a fresh download is fine.
                Log.w(TAG, "downloaded ${role.key} model v$version failed its checksum on load, using bundled model")
                clearInstalled(context, role, rejected = false)
                null
            } else {
                Downloaded(version, bytes)
            }
        } catch (e: Exception) {
            Log.w(TAG, "could not read downloaded ${role.key} model v$version, using bundled model", e)
            clearInstalled(context, role, rejected = false)
            null
        }
    }

    /** Called when a downloaded model fails to load. Deletes it and
     * remembers its version so the same file isn't fetched again; a
     * later, higher version is still accepted. */
    fun discardDownloaded(context: Context, role: ModelRole) = clearInstalled(context, role, rejected = true)

    private fun clearInstalled(context: Context, role: ModelRole, rejected: Boolean) {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val version = prefs.getInt(keyVersion(role), 0)
        File(File(context.filesDir, DIR_NAME), modelFile(role)).delete()
        prefs.edit().apply {
            remove(keyVersion(role))
            remove(keySha(role))
            if (rejected && version > 0) putInt(keyRejected(role), version)
        }.commit()
    }

    private fun clearLegacy(context: Context) {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        File(File(context.filesDir, DIR_NAME), LEGACY_MODEL_FILE).delete()
        if (LEGACY_KEYS.any(prefs::contains)) {
            prefs.edit().apply { LEGACY_KEYS.forEach(::remove) }.commit()
        }
    }

    /** Throws if this session's shapes aren't exactly what FusionEngine
     * feeds and reads for [role] (see ModelChecks). */
    fun requireCompatible(session: OrtSession, role: ModelRole) {
        val inputs = session.inputInfo.values.map { (it.info as TensorInfo).shape }
        val outputs = session.outputInfo.values.map { (it.info as TensorInfo).shape }
        require(ModelChecks.shapesCompatible(inputs, outputs, role)) {
            "${role.key} model shapes ${inputs.map { it.toList() }} -> ${outputs.map { it.toList() }} " +
                "don't match ${ModelChecks.inputShape(role.channels).toList()} -> ${ModelChecks.OUTPUT_SHAPE.toList()}"
        }
    }

    /** One Result per role. Never throws; every failure comes back as
     * Result.Failed, and a failure fetching the manifest is reported for
     * every role. */
    suspend fun checkForUpdate(context: Context): Map<ModelRole, Result> = withContext(Dispatchers.IO) {
        val app = context.applicationContext
        val results = try {
            clearLegacy(app)
            val baseUrl = BuildConfig.MODEL_BASE_URL.trimEnd('/')
            val manifestJson = fetch("$baseUrl/manifest.json", noCache = true, maxBytes = 16 * 1024)
            val manifests = EnsembleManifest.parse(String(manifestJson, Charsets.UTF_8))
            ModelRole.values().associateWith { role ->
                val m = manifests[role] ?: return@associateWith Result.UpToDate
                try {
                    check(app, baseUrl, role, m)
                } catch (e: Exception) {
                    Result.Failed(e.toString())
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "update check failed: $e")
            ModelRole.values().associateWith { Result.Failed(e.toString()) }
        }
        for ((role, result) in results) {
            when (result) {
                is Result.Installed -> Log.i(TAG, "installed ${role.key} model v${result.version}, active from next app start")
                is Result.UpToDate -> Log.i(TAG, "${role.key} model up to date")
                is Result.Failed -> Log.w(TAG, "${role.key} update rejected: ${result.reason}")
            }
        }
        results
    }

    private fun check(context: Context, baseUrl: String, role: ModelRole, manifest: ModelManifest): Result {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val running = maxOf(bundledVersion(role), prefs.getInt(keyVersion(role), 0))
        val rejected = prefs.getInt(keyRejected(role), 0).takeIf { it > 0 }
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
            validate(bytes, role)
        } catch (e: Exception) {
            // A model that downloads fine but can't run here would fail
            // the same way every time, so don't fetch this version again.
            prefs.edit().putInt(keyRejected(role), manifest.version).commit()
            return Result.Failed("v${manifest.version}: ${e.message}")
        }

        // Write to a temp file and rename, so a crash mid-write never
        // leaves a half-written model behind.
        val dir = File(context.filesDir, DIR_NAME).apply { mkdirs() }
        val tmp = File(dir, "${modelFile(role)}.tmp")
        tmp.writeBytes(bytes)
        val target = File(dir, modelFile(role))
        if (!tmp.renameTo(target)) {
            tmp.delete()
            return Result.Failed("v${manifest.version}: could not move model into place")
        }
        prefs.edit()
            .putInt(keyVersion(role), manifest.version)
            .putString(keySha(role), sha)
            .commit()
        return Result.Installed(manifest.version)
    }

    /** Opens the model, checks its shapes and runs one all-zero window,
     * which is a stationary phone. Throws on any problem. */
    private fun validate(bytes: ByteArray, role: ModelRole) {
        val env = OrtEnvironment.getEnvironment()
        env.createSession(bytes, OrtSession.SessionOptions()).use { session ->
            requireCompatible(session, role)
            val window = FloatArray(ModelChecks.WINDOW.toInt() * role.channels)
            OnnxTensor.createTensor(env, FloatBuffer.wrap(window), ModelChecks.inputShape(role.channels)).use { input ->
                session.run(mapOf(session.inputNames.first() to input)).use { result ->
                    @Suppress("UNCHECKED_CAST")
                    val out = (result[0].value as Array<Array<FloatArray>>)[0]
                    require(ModelChecks.outputUsable(out)) {
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
