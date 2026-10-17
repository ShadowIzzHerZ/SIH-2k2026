package com.sih26168.deadreckoning

import org.json.JSONObject
import java.io.File
import java.security.MessageDigest

/**
 * One model's entry in the small JSON file ModelUpdater fetches before
 * deciding whether to download anything. scripts/publish_model.py writes
 * it next to the models in the Supabase Storage "models" bucket:
 *
 *     {"version": 3, "file": "dead_reckoning_model_v3.onnx",
 *      "sha256": "<64 hex chars>", "size_bytes": 860807,
 *      "models": {
 *        "slow": {"version": 2, "file": "model_slow_v2.onnx", "sha256": "...", "size_bytes": 860807},
 *        "fast": {"version": 2, "file": "model_fast_v2.onnx", "sha256": "...", "size_bytes": 861447}
 *      }}
 *
 * The top-level fields are the single model older app versions read; this
 * version only reads "models" (see EnsembleManifest), one entry per
 * ModelRole, each with its own version and checksum. Either role may be
 * missing, which just means nothing newer is published for it.
 *
 * Plain Kotlin apart from org.json, so ModelUpdateTest can check the
 * parsing and the accept/reject rules on the JVM without a device.
 */
data class ModelManifest(
    val version: Int,
    val file: String,
    val sha256: String,
    val sizeBytes: Long,
) {
    companion object {
        /** Anything bigger is not one of our models (the current one is
         * ~0.9 MB). Stops a bad manifest from filling the phone's storage. */
        const val MAX_MODEL_BYTES = 20L * 1024 * 1024

        private val SHA256_HEX = Regex("^[0-9a-f]{64}$")
        // A bare file name in the bucket, never a path or URL, so a manifest
        // can't point the download anywhere else.
        private val FILE_NAME = Regex("^[A-Za-z0-9._-]+\\.onnx$")

        /** Throws IllegalArgumentException (or org.json's JSONException)
         * on anything malformed. */
        fun parse(json: String): ModelManifest = parse(JSONObject(json))

        fun parse(o: JSONObject): ModelManifest {
            val version = o.getInt("version")
            val file = o.getString("file")
            val sha256 = o.getString("sha256").lowercase()
            val sizeBytes = o.getLong("size_bytes")
            require(version > 0) { "version must be positive, got $version" }
            require(FILE_NAME.matches(file) && !file.contains("..")) { "bad model file name: $file" }
            require(SHA256_HEX.matches(sha256)) { "sha256 must be 64 hex chars" }
            require(sizeBytes in 1..MAX_MODEL_BYTES) { "size_bytes out of range: $sizeBytes" }
            return ModelManifest(version, file, sha256, sizeBytes)
        }
    }
}

/**
 * The two networks of the speed-gated ensemble (see EnsembleModel and
 * src/export_ensemble.py). [channels] is the exact input width each one
 * was exported with; [key] is its name in the manifest's "models" object
 * and in on-device file and preference names.
 */
enum class ModelRole(val key: String, val asset: String, val channels: Int) {
    /** augbias_long: plain 6 IMU channels, used below the speed gate. */
    SLOW("slow", "model_slow.onnx", 6),
    /** speedaware: 6 IMU channels plus the chunk's start speed / 10. */
    FAST("fast", "model_fast.onnx", 7);

    companion object {
        fun fromKey(key: String): ModelRole? = values().firstOrNull { it.key == key }
    }
}

/** The per-role part of the manifest. */
object EnsembleManifest {
    /** Roles this app doesn't know are ignored, so a later app version can
     * add one without breaking this one. A known role with a malformed
     * entry throws: something is wrong with the publish, and nothing from
     * this manifest should be trusted. A manifest without "models" (one
     * written before the ensemble existed) gives an empty map. */
    fun parse(json: String): Map<ModelRole, ModelManifest> {
        val models = JSONObject(json).optJSONObject("models") ?: return emptyMap()
        val out = LinkedHashMap<ModelRole, ModelManifest>()
        for (key in models.keys()) {
            val role = ModelRole.fromKey(key) ?: continue
            out[role] = ModelManifest.parse(models.getJSONObject(key))
        }
        return out
    }
}

/** The accept/reject rules for a downloaded model, kept free of Android
 * and ONNX Runtime types so they can be unit tested. */
object ModelChecks {
    // FusionEngine feeds exactly one 5 s window at 10 Hz of
    // [ax, ay, az, gx, gy, gz] (plus the start speed for ModelRole.FAST)
    // and reads [delta_v, delta_theta] per step. A model with any other
    // shape would crash or silently misread, so a download that doesn't
    // match its role's shape exactly is rejected.
    const val WINDOW = 50L
    val INPUT_SHAPE = longArrayOf(1, WINDOW, 6)
    val OUTPUT_SHAPE = longArrayOf(1, WINDOW, 2)

    fun inputShape(channels: Int) = longArrayOf(1, WINDOW, channels.toLong())

    fun shapesCompatible(
        inputShapes: List<LongArray>,
        outputShapes: List<LongArray>,
        channels: Int = 6,
    ): Boolean =
        inputShapes.size == 1 && outputShapes.size == 1 &&
            inputShapes[0].contentEquals(inputShape(channels)) &&
            outputShapes[0].contentEquals(OUTPUT_SHAPE)

    fun shapesCompatible(inputShapes: List<LongArray>, outputShapes: List<LongArray>, role: ModelRole) =
        shapesCompatible(inputShapes, outputShapes, role.channels)

    /** Download only when the manifest offers something newer than what
     * would run now, and not a version that already failed on this phone. */
    fun shouldDownload(manifestVersion: Int, runningVersion: Int, rejectedVersion: Int?): Boolean =
        manifestVersion > runningVersion && manifestVersion != rejectedVersion

    /** A (WINDOW, 2) correction sequence with only finite numbers. */
    fun outputUsable(out: Array<FloatArray>): Boolean =
        out.size.toLong() == WINDOW && out.all { it.size == 2 && it.all(Float::isFinite) }

    fun sha256Hex(bytes: ByteArray): String =
        MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }

    fun sha256Hex(file: File): String = sha256Hex(file.readBytes())
}
