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
     *
     * No longer used by CalibrationManager: it compares phone-frame accel
     * angles with world compass headings, so its answer includes the
     * drive's compass heading. See [estimateForwardAxis].
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

    // ---------------------------------------------------- forward axis

    /**
     * Forward axis (yaw offset) from what GNSS says the vehicle did — the
     * fix from the drift experiments (experiments/drift/lib.py's
     * calibrate_fwdfix, drift-experiments branch), which the ensemble
     * models were trained on. In the leveled frame, horizontal accel
     * rotated by psi should equal [dv/dt, v * yaw_rate] (longitudinal,
     * centripetal), so with both as complex numbers
     *
     *     psi = angle(sum(conj(ax + i*ay) * (dv/dt + i*v*wz)))
     *
     * over samples above [minSpeed], every signal smoothed with
     * [smoothEdgeNormalized]. Unlike [estimateYawMisalignment] it never
     * compares phone-frame angles with compass headings, which mixed the
     * drive's mean compass heading into the estimate.
     *
     * levelAccel/levelGyro are leveled (R_level applied) samples, speed is
     * GNSS speed, all at a fixed [dt]. Returns null below [minSamples]
     * usable samples, where calibrate_fwdfix falls back to 0.
     */
    fun estimateForwardAxis(
        levelAccel: List<FloatArray>,
        levelGyro: List<FloatArray>,
        speed: FloatArray,
        dt: Float = 0.1f,
        minSpeed: Float = FORWARD_AXIS_MIN_SPEED,
        minSamples: Int = FORWARD_AXIS_MIN_SAMPLES,
    ): ForwardAxisFit? {
        val n = speed.size
        require(levelAccel.size == n && levelGyro.size == n) { "accel, gyro and speed must be the same length" }
        val v = DoubleArray(n) { speed[it].toDouble() }
        val vs = smoothEdgeNormalized(v)
        val dvdt = gradient(vs, dt.toDouble())
        val wz = smoothEdgeNormalized(DoubleArray(n) { levelGyro[it][2].toDouble() })
        val ax = smoothEdgeNormalized(DoubleArray(n) { levelAccel[it][0].toDouble() })
        val ay = smoothEdgeNormalized(DoubleArray(n) { levelAccel[it][1].toDouble() })
        val acc = ForwardAxisSum()
        for (i in 0 until n) {
            if (v[i] > minSpeed) acc.add(ax[i], ay[i], dvdt[i], vs[i] * wz[i])
        }
        return if (acc.count > minSamples) acc.fit() else null
    }

    /** numpy's convolve(x, ones(k), "same") / convolve(ones, ones(k),
     * "same"): a k-sample moving average (window [i - k/2, i + k/2 - 1]
     * for even k) that averages over only the samples that exist at the
     * ends, instead of pulling them toward zero. */
    fun smoothEdgeNormalized(x: DoubleArray, k: Int = FORWARD_AXIS_SMOOTH): DoubleArray {
        val n = x.size
        val lo = k / 2
        val hi = k - 1 - lo
        return DoubleArray(n) { i ->
            var sum = 0.0
            var count = 0
            for (j in maxOf(0, i - lo)..minOf(n - 1, i + hi)) {
                sum += x[j]; count++
            }
            sum / count
        }
    }

    /** numpy.gradient: central differences inside, one-sided at the ends. */
    fun gradient(x: DoubleArray, dt: Double): DoubleArray {
        val n = x.size
        if (n < 2) return DoubleArray(n)
        return DoubleArray(n) { i ->
            when (i) {
                0 -> (x[1] - x[0]) / dt
                n - 1 -> (x[n - 1] - x[n - 2]) / dt
                else -> (x[i + 1] - x[i - 1]) / (2 * dt)
            }
        }
    }

    const val FORWARD_AXIS_MIN_SPEED = 3f
    const val FORWARD_AXIS_MIN_SAMPLES = 100
    const val FORWARD_AXIS_SMOOTH = 10
}

/** A forward-axis estimate: [psi] in radians, from [samples] usable
 * samples. [coherence] is |sum| / sum(|term|), 1 when every sample agrees
 * on the direction and near 0 for noise — a real drive measures ~0.9,
 * pure sensor noise ~0.05. */
data class ForwardAxisFit(val psi: Float, val samples: Int, val coherence: Float)

/** The running complex sum behind [Calibration.estimateForwardAxis]. */
class ForwardAxisSum {
    private var re = 0.0
    private var im = 0.0
    private var absSum = 0.0
    var count = 0
        private set

    /** One sample: smoothed leveled accel (ax, ay) and the target
     * (dv/dt, v * yaw_rate). Adds conj(ax + i*ay) * (lon + i*lat). */
    fun add(ax: Double, ay: Double, lon: Double, lat: Double) {
        val pr = ax * lon + ay * lat
        val pi = ax * lat - ay * lon
        if (!pr.isFinite() || !pi.isFinite()) return
        re += pr
        im += pi
        absSum += sqrt(pr * pr + pi * pi)
        count++
    }

    fun fit(): ForwardAxisFit = ForwardAxisFit(
        psi = atan2(im, re).toFloat(),
        samples = count,
        coherence = if (absSum > 0) (sqrt(re * re + im * im) / absSum).toFloat() else 0f,
    )
}

/**
 * The same estimate fed one sample at a time, for live calibration, in
 * the raw phone frame so the leveling can keep improving while the drive
 * goes on (see CalibrationManager): the sum the fit needs is bilinear in
 * the leveling rotation, so it is kept as a 3-vector P = sum(a * dv/dt)
 * and a 3x3 Q = sum(a * (v * w)^T) over smoothed raw accel a and raw gyro
 * w, and [fit] applies whichever R_level is current. With R_level
 * x/y/z rows r0, r1, r2:
 *
 *     re = r0.P + r1 Q r2,   im = r0 Q r2 - r1.P
 *
 * Each sample is processed once the samples it needs on both sides have
 * arrived (smoothing needs FORWARD_AXIS_SMOOTH / 2 ahead, dv/dt one more),
 * so the sum always covers the whole drive so far. [gap] ends a
 * contiguous stretch (GNSS lost): samples are never smoothed across it,
 * and the few at its edges are dropped instead of edge-normalised, the
 * only difference from the batch version.
 */
class ForwardAxisEstimator(
    private val dt: Float = 0.1f,
    private val minSpeed: Float = Calibration.FORWARD_AXIS_MIN_SPEED,
    private val k: Int = Calibration.FORWARD_AXIS_SMOOTH,
) {
    private val p = DoubleArray(3)
    private val q = DoubleArray(9)
    // sum(|term|) for the coherence, under the leveling current when each
    // sample was added (not bilinear, so it can't be re-leveled).
    private var absSum = 0.0
    var samples = 0
        private set

    // [ax, ay, az, gx, gy, gz, v] per raw sample, the last `need` of them.
    private val buf = ArrayDeque<DoubleArray>()
    private val lo = k / 2
    private val hi = k - 1 - lo
    // Center sample c needs smoothed v at c-1 and c+1, so raw samples
    // c-1-lo .. c+1+hi.
    private val need = lo + hi + 3

    /** [rLevel] only weighs this sample's share of the coherence. */
    fun add(rawAccel: FloatArray, rawGyro: FloatArray, speed: Float, rLevel: FloatArray) {
        buf.addLast(
            doubleArrayOf(
                rawAccel[0].toDouble(), rawAccel[1].toDouble(), rawAccel[2].toDouble(),
                rawGyro[0].toDouble(), rawGyro[1].toDouble(), rawGyro[2].toDouble(), speed.toDouble(),
            )
        )
        if (buf.size > need) buf.removeFirst()
        if (buf.size < need) return
        val c = lo + 1  // index of the center sample within buf
        fun mean(col: Int, center: Int): Double {
            var s = 0.0
            for (j in center - lo..center + hi) s += buf[j][col]
            return s / k
        }
        if (buf[c][6] <= minSpeed) return
        val vs = mean(6, c)
        val dvdt = (mean(6, c + 1) - mean(6, c - 1)) / (2 * dt)
        val a = DoubleArray(3) { mean(it, c) }
        val l = DoubleArray(3) { vs * mean(3 + it, c) }
        if (!(a.all(Double::isFinite) && l.all(Double::isFinite) && dvdt.isFinite())) return
        for (i in 0 until 3) {
            p[i] += a[i] * dvdt
            for (j in 0 until 3) q[i * 3 + j] += a[i] * l[j]
        }
        val ax = dot(rLevel, 0, a); val ay = dot(rLevel, 1, a); val lat = dot(rLevel, 2, l)
        val pr = ax * dvdt + ay * lat
        val pi = ax * lat - ay * dvdt
        absSum += sqrt(pr * pr + pi * pi)
        samples++
    }

    fun gap() = buf.clear()

    /** Null until more than [minSamples] samples have counted. */
    fun fit(rLevel: FloatArray, minSamples: Int = Calibration.FORWARD_AXIS_MIN_SAMPLES): ForwardAxisFit? {
        if (samples <= minSamples) return null
        val qr2 = DoubleArray(3) { i -> (0 until 3).sumOf { j -> q[i * 3 + j] * rLevel[6 + j] } }
        val re = dot(rLevel, 0, p) + dot(rLevel, 1, qr2)
        val im = dot(rLevel, 0, qr2) - dot(rLevel, 1, p)
        val mag = sqrt(re * re + im * im)
        return ForwardAxisFit(
            psi = atan2(im, re).toFloat(),
            samples = samples,
            coherence = if (absSum > 0) minOf(1.0, mag / absSum).toFloat() else 0f,
        )
    }

    private fun dot(r: FloatArray, row: Int, v: DoubleArray) =
        r[row * 3] * v[0] + r[row * 3 + 1] * v[1] + r[row * 3 + 2] * v[2]
}
