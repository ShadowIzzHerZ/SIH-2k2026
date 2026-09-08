package com.sih26168.deadreckoning

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import java.nio.FloatBuffer

/**
 * Loads and runs the exported BiasCorrectionNet (checkpoints/
 * dead_reckoning_model.onnx, bundled as an asset by the app module's
 * syncModel Gradle task) via ONNX Runtime Mobile.
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
class BiasCorrectionModel(context: Context, assetName: String = "dead_reckoning_model.onnx") {

    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        val modelBytes = context.assets.open(assetName).use { it.readBytes() }
        session = env.createSession(modelBytes, OrtSession.SessionOptions())
    }

    /**
     * window: T samples of [ax, ay, az, gx, gy, gz] (calibrated, vehicle
     * frame). Returns T samples of [delta_v, delta_theta].
     */
    fun predict(window: Array<FloatArray>): Array<FloatArray> {
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
}
