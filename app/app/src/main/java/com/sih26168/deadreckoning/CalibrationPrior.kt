package com.sih26168.deadreckoning

import org.json.JSONArray
import org.json.JSONObject

/**
 * What a previous drive learned about how the phone is mounted: the
 * drive-long gravity direction (unit vector, raw phone frame) and the
 * forward axis psi that went with it, from [samples] usable samples.
 * CalibrationManager starts from it when the phone is back in the same
 * mount (see its `prior` parameter), so a short drive doesn't have to
 * relearn the forward axis from scratch. Plain Kotlin + org.json so the
 * JSON round trip is unit tested; CalibrationPriorStore keeps it in
 * SharedPreferences.
 */
data class CalibrationPrior(val gravity: FloatArray, val psi: Float, val samples: Int) {

    init {
        require(gravity.size == 3) { "gravity must be a 3-vector" }
    }

    fun toJson(): String = JSONObject()
        .put("version", FORMAT_VERSION)
        .put("gravity", JSONArray().apply { gravity.forEach { put(it.toDouble()) } })
        .put("psi", psi.toDouble())
        .put("samples", samples)
        .toString()

    override fun equals(other: Any?) =
        other is CalibrationPrior && gravity.contentEquals(other.gravity) && psi == other.psi && samples == other.samples

    override fun hashCode() = (gravity.contentHashCode() * 31 + psi.hashCode()) * 31 + samples

    companion object {
        const val FORMAT_VERSION = 1

        /** Null for anything malformed, from another format version, or
         * not physically meaningful, so a bad saved value just means no
         * warm start. */
        fun fromJson(json: String?): CalibrationPrior? {
            if (json == null) return null
            return try {
                parse(JSONObject(json))
            } catch (e: Exception) {
                null
            }
        }

        private fun parse(o: JSONObject): CalibrationPrior? {
            if (o.getInt("version") != FORMAT_VERSION) return null
            val g = o.getJSONArray("gravity")
            if (g.length() != 3) return null
            val gravity = FloatArray(3) { g.getDouble(it).toFloat() }
            val norm = kotlin.math.sqrt(gravity.sumOf { (it * it).toDouble() })
            val psi = o.getDouble("psi").toFloat()
            val samples = o.getInt("samples")
            return if (!gravity.all(Float::isFinite) || norm < 0.5 || norm > 1.5 || !psi.isFinite() || samples <= 0) null
            else CalibrationPrior(FloatArray(3) { (gravity[it] / norm).toFloat() }, Calibration.wrapAngle(psi), samples)
        }
    }
}
