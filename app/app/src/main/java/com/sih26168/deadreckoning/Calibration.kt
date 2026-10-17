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
     * Estimate the constant yaw offset between the leveled phone frame and
     * the vehicle forward direction, against GPS speed and course — same
     * method as calibration.py's estimate_yaw_misalignment.
     *
     * The leveled phone frame has an unknown yaw, so its accel direction
     * can't be compared with a world-frame course directly (that used to
     * return the mount offset plus the average course). Instead both sides
     * are put in the vehicle frame: GPS says the car had longitudinal accel
     * dv/dt and lateral accel v * d(course)/dt, and psi is the
     * least-squares 2D rotation taking the measured horizontal accel onto
     * that. Braking counts the same as accelerating instead of voting for
     * a 180 degree flip.
     *
     * levelAccelXY, gtHeadingXY (unit direction-of-travel vectors, any
     * fixed world frame) and speedMps are consecutive samples dtS apart.
     * Samples slower than minSpeedMps, or whose GPS-implied accel is below
     * minHorizAccel m/s^2 (no direction), are ignored. Returns 0 (i.e.
     * uncalibrated) rather than a guess when nothing is usable or the fit
     * quality (see [yawMisalignmentSpread]) is below minFitQuality.
     */
    fun estimateYawMisalignment(
        levelAccelXY: List<FloatArray>, gtHeadingXY: List<FloatArray>, speedMps: List<Float>, dtS: Float,
        minSpeedMps: Float = 2.0f, minHorizAccel: Float = 0.3f, minFitQuality: Float = 0.5f,
    ): Float {
        val fit = yawFit(levelAccelXY, gtHeadingXY, speedMps, dtS, minSpeedMps, minHorizAccel) ?: return 0f
        if (fit.quality < minFitQuality) return 0f
        return atan2(fit.cross, fit.dot).toFloat()
    }

    /**
     * How much the samples behind [estimateYawMisalignment] actually agree
     * on one angle: 1 - |sum m.e| / sum |m||e| over the usable samples
     * (m = measured, e = GPS-implied accel). 0 = every sample gives the
     * same rotation, approaching 1 = no consistent rotation; 1 when no
     * sample is usable at all.
     *
     * NOT part of the calibration.py port; added for *live* calibration,
     * where the yaw estimate locks in once at trip start and everything
     * after it inherits the error (see CalibrationManager).
     */
    fun yawMisalignmentSpread(
        levelAccelXY: List<FloatArray>, gtHeadingXY: List<FloatArray>, speedMps: List<Float>, dtS: Float,
        minSpeedMps: Float = 2.0f, minHorizAccel: Float = 0.3f,
    ): Float {
        val fit = yawFit(levelAccelXY, gtHeadingXY, speedMps, dtS, minSpeedMps, minHorizAccel) ?: return 1f
        return (1.0 - fit.quality).toFloat()
    }

    private class YawFit(val cross: Double, val dot: Double, val quality: Double)

    /** The least-squares rotation sums both of the above are built on, over
     * the usable samples — shared so the estimate and its spread can never
     * be computed over different sample sets. Null when no sample is
     * usable. */
    private fun yawFit(
        levelAccelXY: List<FloatArray>, gtHeadingXY: List<FloatArray>, speedMps: List<Float>, dtS: Float,
        minSpeedMps: Float, minHorizAccel: Float,
    ): YawFit? {
        val n = minOf(levelAccelXY.size, gtHeadingXY.size, speedMps.size)
        if (n < 3) return null
        val crs = DoubleArray(n) { atan2(gtHeadingXY[it][1].toDouble(), gtHeadingXY[it][0].toDouble()) }
        var cross = 0.0
        var dot = 0.0
        var norm = 0.0
        for (i in 1 until n - 1) {
            val v = speedMps[i].toDouble()
            // Central differences: GPS-implied [forward, left] accel in the vehicle frame.
            val eLon = (speedMps[i + 1] - speedMps[i - 1]) / (2.0 * dtS)
            val dCrs = crs[i + 1] - crs[i - 1]
            val eLat = v * atan2(sin(dCrs), cos(dCrs)) / (2.0 * dtS)
            val mx = levelAccelXY[i][0].toDouble()
            val my = levelAccelXY[i][1].toDouble()
            if (!(eLon.isFinite() && eLat.isFinite() && mx.isFinite() && my.isFinite())) continue
            if (!(v > minSpeedMps)) continue
            val eMag = sqrt(eLon * eLon + eLat * eLat)
            if (eMag <= minHorizAccel) continue
            cross += mx * eLat - my * eLon
            dot += mx * eLon + my * eLat
            norm += sqrt(mx * mx + my * my) * eMag
        }
        if (norm <= 0.0) return null
        return YawFit(cross, dot, sqrt(cross * cross + dot * dot) / norm)
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

    /**
     * Compass bearing (degrees, 0=N/90=E/clockwise — what Location.getBearing()
     * and this app's replay/DevRecorder CSVs both store) -> this app's own
     * internal heading convention (radians, 0=east, ccw+ — FusionState.heading,
     * MapMatcher.match's headingRad, strapdown_ins.py's theta all agree on
     * this). NOT a plain degrees->radians unit conversion — compass bearing is
     * measured clockwise from north, this app's headings counterclockwise
     * from east, so the axes have to be swapped, not just the units.
     *
     * Real bug this fixes: MainActivity.readSample() and ReplayDataSource
     * both used to do only Math.toRadians(bearing) — degrees->radians, no
     * axis swap — then hand that straight to FusionEngine.tick's
     * gnssHeadingRad (assigned directly to state.heading) and to
     * CalibrationManager.addYawSample (which itself correctly assumes an
     * already-math-convention input). The result: chunkAnchor.heading at
     * the start of a blackout, and the yaw-misalignment estimate underlying
     * every calibrated sample all session, were both silently off by a
     * heading-dependent amount (compass->math is a reflection, not a fixed
     * rotation) — reported live as the blackout trail heading off in a very
     * different direction from the real one. ReplayDataSource's own
     * bearingRad doc ("already converted from compass degrees") describes
     * the intent this function actually implements; the plain toRadians()
     * call there never did.
     */
    fun compassDegToMathRad(bearingDeg: Float): Float {
        val unit = compassDegToXyUnit(bearingDeg)  // [east, north]
        return atan2(unit[1].toDouble(), unit[0].toDouble()).toFloat()
    }

    /** Inverse of compassDegToMathRad — this app's internal heading (radians,
     * 0=east ccw+) -> compass bearing degrees (0=N, 90=E, clockwise), in
     * [0, 360). For anything that has to show or store a real-world compass
     * heading from a FusionState.heading value: the live UI's heading
     * readout, and DevRecorder's heading_gt column (src/data/io_vnbd_loader.py
     * and calibration.py/CalibrationManager both expect heading_gt in this
     * same compass convention, not this app's internal math one). */
    fun mathRadToCompassDeg(mathRad: Float): Float {
        val east = cos(mathRad.toDouble())
        val north = sin(mathRad.toDouble())
        val deg = Math.toDegrees(atan2(east, north)).toFloat()
        return if (deg < 0) deg + 360f else deg
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
