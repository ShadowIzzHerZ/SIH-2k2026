package com.sih26168.deadreckoning

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import java.nio.FloatBuffer

/**
 * FusionEngine's only real dependency on BiasCorrectionModel — pulled out
 * so a plain JVM unit test (no device, no ONNX runtime, no Android
 * Context) can exercise FusionEngine's actual integration/ZUPT logic
 * against a fake Predictor instead. See FusionEngineTest.
 */
interface Predictor {
    fun predict(window: Array<FloatArray>): Array<FloatArray>
}

/**
 * Loads and runs the exported BiasCorrectionNet (checkpoints/
 * dead_reckoning_model.onnx, bundled as an asset by the app module's
 * syncModel Gradle task, or a newer verified copy fetched by
 * ModelUpdater) via ONNX Runtime Mobile.
 *
 * Contract matches src/export_onnx.py's docstring exactly: a
 * (1, T, 6) calibrated-IMU window in — [ax, ay, az, gx, gy, gz], vehicle
 * frame, real physical units, NOT z-score normalized (this checkpoint is
 * the plain 6-raw-channel production model, not the reverted
 * --extra_features variant — see configs/default.yaml) — producing a
 * (1, T, 2) [delta_v, delta_theta] correction sequence out. T is
 * whatever's fed in; the ONNX graph itself was exported with a fixed
 * window_size (see export_onnx.py), matching FusionEngine's chunking.
 */
class BiasCorrectionModel(context: Context, assetName: String = DEFAULT_ASSET) : Predictor {

    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    /** Which model is running: ModelUpdater.BUNDLED_MODEL_VERSION for the
     * APK's own copy, or the version of a verified download. */
    val loadedVersion: Int

    init {
        // Prefer a model ModelUpdater downloaded and verified on an earlier
        // run. Anything wrong with it (fails to open, wrong shapes) means
        // it is deleted and the bundled model loads instead, so an update
        // can never leave the app without a working model.
        var downloadedSession: OrtSession? = null
        var version = ModelUpdater.BUNDLED_MODEL_VERSION
        val downloaded = if (assetName == DEFAULT_ASSET) ModelUpdater.downloadedModel(context) else null
        if (downloaded != null) {
            try {
                downloadedSession = env.createSession(downloaded.bytes, OrtSession.SessionOptions())
                ModelUpdater.requireCompatible(downloadedSession)
                version = downloaded.version
            } catch (e: Exception) {
                Log.w(TAG, "downloaded model v${downloaded.version} failed to load, using bundled model", e)
                downloadedSession?.close()
                downloadedSession = null
                ModelUpdater.discardDownloaded(context)
            }
        }
        session = downloadedSession ?: run {
            val modelBytes = context.assets.open(assetName).use { it.readBytes() }
            env.createSession(modelBytes, OrtSession.SessionOptions())
        }
        loadedVersion = version
        Log.i(TAG, "loaded model v$loadedVersion (${if (downloadedSession != null) "downloaded" else "bundled"})")
    }

    /**
     * window: T samples of [ax, ay, az, gx, gy, gz] (calibrated, vehicle
     * frame). Returns T samples of [delta_v, delta_theta].
     */
    override fun predict(window: Array<FloatArray>): Array<FloatArray> {
        val t = window.size
        val flat = FloatArray(t * 6)
        for (i in 0 until t) {
            System.arraycopy(window[i], 0, flat, i * 6, 6)
        }
        val shape = longArrayOf(1, t.toLong(), 6)
        OnnxTensor.createTensor(env, FloatBuffer.wrap(flat), shape).use { inputTensor ->
            val inputName = session.inputNames.iterator().next()
            session.run(mapOf(inputName to inputTensor)).use { result ->
                @Suppress("UNCHECKED_CAST")
                val out = result[0].value as Array<Array<FloatArray>>
                return out[0]  // (1, T, 2) -> (T, 2)
            }
        }
    }

    fun close() {
        session.close()
    }

    companion object {
        const val DEFAULT_ASSET = "dead_reckoning_model.onnx"
        private const val TAG = "BiasCorrectionModel"
    }
}
