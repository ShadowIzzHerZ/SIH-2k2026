package com.sih26168.deadreckoning

import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Phone-to-vehicle frame calibration/alignment — direct Kotlin port of
 * src/calibration.py, kept behaviorally identical (same two-stage
 * leveling + yaw approach, same reasoning) so the on-device pipeline
 * matches what was trained and validated in Python, not a reimplementation
 * that might subtly diverge.
 *
 * A 3x3 rotation matrix here is a flat 9-element FloatArray, row-major
 * (m[0..2] = row 0, m[3..5] = row 1, m[6..8] = row 2) — no matrix library
 * dependency needed for something this small.
 */
object Calibration {

    /** Estimate the gravity direction from a (mostly) stationary accel window. */
    fun estimateGravityVector(accelSamples: List<FloatArray>): FloatArray {
        val g = floatArrayOf(0f, 0f, 0f)
        for (a in accelSamples) {
            g[0] += a[0]; g[1] += a[1]; g[2] += a[2]
        }
        val n = accelSamples.size.toFloat()
        g[0] /= n; g[1] /= n; g[2] /= n
        val norm = sqrt((g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).toDouble()).toFloat()
        require(norm >= 1e-6f) { "Degenerate gravity estimate — accel window is all zero." }
        return floatArrayOf(g[0] / norm, g[1] / norm, g[2] / norm)
    }

    /**
     * Rotation matrix mapping the phone frame to a level frame (z-axis
     * aligned with true vertical), leaving yaw unresolved — Rodrigues'
     * rotation formula mapping gravityPhone -> [0,0,1], same as
     * calibration.py's leveling_rotation.
     */
    fun levelingRotation(gravityPhone: FloatArray): FloatArray {
        val norm = sqrt(
            (gravityPhone[0] * gravityPhone[0] + gravityPhone[1] * gravityPhone[1] +
                gravityPhone[2] * gravityPhone[2]).toDouble()
        ).toFloat()
        val v = floatArrayOf(gravityPhone[0] / norm, gravityPhone[1] / norm, gravityPhone[2] / norm)
        val target = floatArrayOf(0f, 0f, 1f)

        // axis = v cross target
        val axis = floatArrayOf(
            v[1] * target[2] - v[2] * target[1],
            v[2] * target[0] - v[0] * target[2],
            v[0] * target[1] - v[1] * target[0],
        )
        val s = sqrt((axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).toDouble()).toFloat()
        val c = v[0] * target[0] + v[1] * target[1] + v[2] * target[2]

        if (s < 1e-8f) {
            return if (c > 0) identity3x3() else floatArrayOf(1f, 0f, 0f, 0f, -1f, 0f, 0f, 0f, -1f)
        }
        val ax = axis[0] / s; val ay = axis[1] / s; val az = axis[2] / s
        // K = skew-symmetric cross-product matrix of the (unit) axis
        val k = floatArrayOf(
            0f, -az, ay,
            az, 0f, -ax,
            -ay, ax, 0f,
        )
        val kk = matmul3x3(k, k)
        // R = I + K*s + K^2*(1-c)
        val r = FloatArray(9)
        val id = identity3x3()
        for (i in 0 until 9) {
            r[i] = id[i] + k[i] * s + kk[i] * (1 - c)
        }
        return r
    }

    /** Rotation about z by yaw angle psi (radians). */
    fun yawRotation(psi: Float): FloatArray {
        val c = cos(psi.toDouble()).toFloat()
        val s = sin(psi.toDouble()).toFloat()
        return floatArrayOf(
            c, -s, 0f,
            s, c, 0f,
            0f, 0f, 1f,
        )
    }

    /**
     * Estimate constant yaw offset between the leveled phone frame and the
     * vehicle forward direction, using confident (moving) samples where
     * horizontal accel direction can be compared against GPS course-over-
     * ground — same approach and same reasoning as
     * calibration.py's estimate_yaw_misalignment (median of angle diffs
     * over the highest-magnitude samples, not a full least-squares fit —
     * robust to the low-magnitude samples where direction is ill-defined).
     */
    fun estimateYawMisalignment(levelAccelXY: List<FloatArray>, gtHeadingXY: List<FloatArray>): Float {
        val diffs = yawAngleDiffs(levelAccelXY, gtHeadingXY)
        if (diffs.isEmpty()) return 0f
        return diffs.sorted()[diffs.size / 2]
    }

    /**
     * How much the samples behind [estimateYawMisalignment] actually agree
     * with each other: circular variance (1 - mean resultant length) of
     * the same filtered angle diffs. 0 = perfectly aligned, approaching
     * 1 = scattered all around the circle. Roughly, an angular spread of
     * sigma radians gives sigma^2 / 2.
     *
     * NOT part of the calibration.py port; added for *live* calibration.
     * Offline, a bad window can just be re-run over different data. Live,
     * the yaw estimate locks in once at trip start and everything after it
     * inherits the error, so the initial stretch has to be checked for
     * actually being the straight line the estimate assumes. The median
     * alone can't reveal that: a stretch that swings between two headings
     * (a turn, a lane change, stop-and-go) yields a perfectly *stable*
     * median of two disagreeing populations.
     *
     * Circular variance specifically, rather than a median-absolute-
     * deviation: MAD is robust to a minority of outliers, which is the
     * exact opposite of what's wanted here. Found the hard way in
     * CalibrationManagerTest — after the top-quartile magnitude filter
     * leaves only a handful of samples, a genuinely bimodal stretch that
     * happens to split 8-2 has an MAD of ~0 and sails through the gate.
     * Circular variance is *sensitive* to that minority, and handles
     * angle wrapping natively instead of by hand.
     */
    fun yawMisalignmentSpread(levelAccelXY: List<FloatArray>, gtHeadingXY: List<FloatArray>): Float {
        val diffs = yawAngleDiffs(levelAccelXY, gtHeadingXY)
        if (diffs.size < 2) return 1f
        var sumCos = 0.0
        var sumSin = 0.0
        for (d in diffs) {
            sumCos += cos(d.toDouble())
            sumSin += sin(d.toDouble())
        }
        val n = diffs.size
        val resultant = sqrt((sumCos / n) * (sumCos / n) + (sumSin / n) * (sumSin / n))
        return (1.0 - resultant).toFloat()
    }

    /** The per-sample (heading - accel direction) angle diffs both of the
     * above are built on, restricted to the highest-magnitude quarter of
     * samples — direction is ill-defined where horizontal accel is tiny.
     * Shared so the estimate and its spread can never be computed over
     * different sample sets. */
    private fun yawAngleDiffs(levelAccelXY: List<FloatArray>, gtHeadingXY: List<FloatArray>): List<Float> {
        val mags = levelAccelXY.map { sqrt((it[0] * it[0] + it[1] * it[1]).toDouble()).toFloat() }
        val sorted = mags.sorted()
        val p75 = sorted[(sorted.size * 0.75).toInt().coerceIn(0, sorted.size - 1)]
        val diffs = mutableListOf<Float>()
        for (i in levelAccelXY.indices) {
            if (mags[i] <= p75) continue
            val aAng = atan2(levelAccelXY[i][1].toDouble(), levelAccelXY[i][0].toDouble())
            val gAng = atan2(gtHeadingXY[i][1].toDouble(), gtHeadingXY[i][0].toDouble())
            val diff = atan2(sin(gAng - aAng), cos(gAng - aAng))
            diffs.add(diff.toFloat())
        }
        return diffs
    }

    private fun wrap(a: Float): Float = atan2(sin(a.toDouble()), cos(a.toDouble())).toFloat()

    /** Apply the full phone->vehicle rotation to one raw IMU sample. */
    fun calibrateSample(raw: FloatArray, rLevel: FloatArray, psiYaw: Float): FloatArray {
        val r = matmul3x3(yawRotation(psiYaw), rLevel)
        return matVec3(r, raw)
    }

    /** Compass bearing (0=N, 90=E, clockwise) -> unit vector [east, north]. */
    fun compassDegToXyUnit(bearingDeg: Float): FloatArray {
        val rad = Math.toRadians(bearingDeg.toDouble())
        return floatArrayOf(sin(rad).toFloat(), cos(rad).toFloat())
    }

    private fun identity3x3() = floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f)

    private fun matmul3x3(a: FloatArray, b: FloatArray): FloatArray {
        val r = FloatArray(9)
        for (i in 0 until 3) for (j in 0 until 3) {
            var s = 0f
            for (k in 0 until 3) s += a[i * 3 + k] * b[k * 3 + j]
            r[i * 3 + j] = s
        }
        return r
    }

    fun matVec3(m: FloatArray, v: FloatArray): FloatArray {
        return floatArrayOf(
            m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
            m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
            m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
        )
    }
}
