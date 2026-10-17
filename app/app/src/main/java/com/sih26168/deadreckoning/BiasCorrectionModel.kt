package com.sih26168.deadreckoning

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import java.nio.FloatBuffer

/**
 * Loads and runs one role of the exported ensemble (checkpoints/
 * model_slow.onnx or model_fast.onnx, bundled as assets by the app
 * module's syncModel Gradle task, or a newer verified copy fetched by
 * ModelUpdater) via ONNX Runtime Mobile.
 *
 * Contract matches src/export_ensemble.py's docstring exactly: a
 * (1, 50, C) window in, C = role.channels — [ax, ay, az, gx, gy, gz],
 * vehicle frame, real physical units, NOT z-score normalized, plus for
 * ModelRole.FAST the start speed / 10 as a 7th channel (EnsembleModel
 * adds it) — producing a (1, 50, 2) [delta_v, delta_theta] correction
 * sequence out. The graph has a fixed window size, matching
 * FusionEngine's chunking.
 */
class BiasCorrectionModel private constructor(
    private val env: OrtEnvironment,
    private val session: OrtSession,
    val role: ModelRole,
    /** ModelUpdater.bundledVersion(role) for the APK's own copy, or the
     * version of a verified download. */
    val loadedVersion: Int,
    val source: ModelSource,
) : Predictor, AutoCloseable {

    /**
     * window: 50 samples of role.channels values. Returns 50 samples of
     * [delta_v, delta_theta].
     */
    override fun predict(window: Array<FloatArray>): Array<FloatArray> {
        val t = window.size
        val c = role.channels
        val flat = FloatArray(t * c)
        for (i in 0 until t) {
            require(window[i].size == c) { "${role.key} model takes $c channels, got ${window[i].size}" }
            System.arraycopy(window[i], 0, flat, i * c, c)
        }
        val shape = longArrayOf(1, t.toLong(), c.toLong())
        OnnxTensor.createTensor(env, FloatBuffer.wrap(flat), shape).use { inputTensor ->
            val inputName = session.inputNames.iterator().next()
            session.run(mapOf(inputName to inputTensor)).use { result ->
                @Suppress("UNCHECKED_CAST")
                val out = result[0].value as Array<Array<FloatArray>>
                return out[0]  // (1, T, 2) -> (T, 2)
            }
        }
    }

    override fun close() {
        session.close()
    }

    companion object {
        private const val TAG = "BiasCorrectionModel"

        /**
         * Prefers a model ModelUpdater downloaded and verified on an
         * earlier run. Anything wrong with it (fails to open, wrong
         * shapes) means it is deleted and the bundled model loads instead,
         * so an update can never leave the app without a working model.
         * Throws only if the bundled model for [role] is broken too.
         */
        fun load(context: Context, role: ModelRole): BiasCorrectionModel {
            val env = OrtEnvironment.getEnvironment()
            val downloaded = ModelUpdater.downloadedModel(context, role)
            val loaded = ModelFallback.load(
                downloadedVersion = downloaded?.version,
                downloadedBytes = downloaded?.let { d -> { d.bytes } },
                bundledVersion = ModelUpdater.bundledVersion(role),
                bundledBytes = { context.assets.open(role.asset).use { it.readBytes() } },
                open = { bytes -> env.createSession(bytes, OrtSession.SessionOptions()) },
                check = { ModelUpdater.requireCompatible(it, role) },
                close = { it.close() },
                onDownloadedFailed = { e ->
                    Log.w(TAG, "downloaded ${role.key} model v${downloaded?.version} failed to load, using bundled model", e)
                    ModelUpdater.discardDownloaded(context, role)
                },
            )
            Log.i(TAG, "loaded ${role.key} model v${loaded.version} (${loaded.source.name.lowercase()})")
            return BiasCorrectionModel(env, loaded.session, role, loaded.version, loaded.source)
        }

        /**
         * Both roles as one EnsembleModel. A role that can't load even from
         * the APK is left out and the other covers every speed; only if
         * both fail does this throw.
         */
        fun loadEnsemble(context: Context): EnsembleModel {
            val models = ModelRole.values().associateWith { role ->
                try {
                    load(context, role)
                } catch (e: Exception) {
                    Log.e(TAG, "${role.key} model failed to load even from the APK; the other model covers every speed", e)
                    null
                }
            }
            return EnsembleModel(slow = models[ModelRole.SLOW], fast = models[ModelRole.FAST])
        }
    }
}
