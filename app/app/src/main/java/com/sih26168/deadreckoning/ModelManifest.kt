package com.sih26168.deadreckoning

import org.json.JSONObject
import java.io.File
import java.security.MessageDigest

/**
 * The small JSON file ModelUpdater fetches before deciding whether to
 * download anything. scripts/publish_model.py writes it next to the model
 * in the Supabase Storage "models" bucket:
 *
 *     {"version": 3, "file": "dead_reckoning_model_v3.onnx",
 *      "sha256": "<64 hex chars>", "size_bytes": 860807}
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
        fun parse(json: String): ModelManifest {
            val o = JSONObject(json)
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

/** The accept/reject rules for a downloaded model, kept free of Android
 * and ONNX Runtime types so they can be unit tested. */
object ModelChecks {
    // FusionEngine feeds exactly one 5 s window at 10 Hz of
    // [ax, ay, az, gx, gy, gz] and reads [delta_v, delta_theta] per step.
    // A model with any other shape would crash or silently misread, so a
    // download that doesn't match this exactly is rejected.
    val INPUT_SHAPE = longArrayOf(1, 50, 6)
    val OUTPUT_SHAPE = longArrayOf(1, 50, 2)

    fun shapesCompatible(inputShapes: List<LongArray>, outputShapes: List<LongArray>): Boolean =
        inputShapes.size == 1 && outputShapes.size == 1 &&
            inputShapes[0].contentEquals(INPUT_SHAPE) &&
            outputShapes[0].contentEquals(OUTPUT_SHAPE)

    /** Download only when the manifest offers something newer than what
     * would run now, and not a version that already failed on this phone. */
    fun shouldDownload(manifestVersion: Int, runningVersion: Int, rejectedVersion: Int?): Boolean =
        manifestVersion > runningVersion && manifestVersion != rejectedVersion

    fun sha256Hex(bytes: ByteArray): String =
        MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }

    fun sha256Hex(file: File): String = sha256Hex(file.readBytes())
}
