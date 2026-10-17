package com.sih26168.deadreckoning

import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Drives Calibration.kt's two-stage leveling + yaw estimate live, on
 * startup, instead of over an already-recorded file — same two stages as
 * calibration.py's module docstring: leveling first (assumes the phone is
 * roughly still for the first `levelSamples` readings, e.g. mounted before
 * pulling off), then yaw alignment once during an initial confident
 * (moving, GPS-available) stretch, then refined for the rest of the
 * drive while GNSS is available.
 *
 * minYawSamples is the offline fit's own floor (100 samples above 3 m/s,
 * 10 s at 10 Hz); a live app can't make the user drive for minutes before
 * showing anything, and the estimate keeps improving after lock-in anyway.
 *
 * Both stages are *gated on the data actually being usable*, rather than
 * on a sample count alone — the difference between offline calibration
 * (where a bad window can be re-run) and live calibration (where a bad
 * estimate silently poisons every downstream number for the rest of the
 * trip, with nothing to catch it):
 *
 *   - Leveling waits for a window whose *orientation* held steady, not
 *     just the first `levelSamples` readings. Whatever gravity direction
 *     it estimates becomes the phone->vehicle leveling rotation for the
 *     entire session, so accepting a window recorded while the phone was
 *     still being picked up/mounted tilts every subsequent calibrated
 *     accel/gyro sample — and nothing downstream can tell that happened.
 *     A sliding window + steadiness check means "hold the phone still"
 *     is verified rather than assumed, and self-corrects (keeps sliding)
 *     if the user is still fumbling with it. That window is only the
 *     starting point: while GNSS is available the gravity estimate keeps
 *     averaging over the whole drive (see relevel), as the offline
 *     calibration the models were trained on does. Steadiness of direction
 *     rather than of |accel|, so a phone already mounted in a moving
 *     vehicle levels immediately instead of waiting out the timeout.
 *   - Yaw comes from what GNSS says the vehicle did (speed change and
 *     turning, see Calibration.estimateForwardAxis — the fixed
 *     alignment the ensemble models were trained on), accumulated over
 *     the whole drive so far. It locks in once the estimate has settled
 *     — [yawRequiredAgreements] checkpoints in a row agreeing — and the
 *     samples behind it agree with each other (coherence at least
 *     [minYawCoherence]): a steady straight cruise has nothing to fit
 *     and must not lock in on noise. After lock-in the sum keeps growing
 *     while GNSS is available and psiYaw follows it, so the live
 *     estimate converges on the same whole-drive answer the offline
 *     calibration gives.
 *
 *     The earlier yaw estimate (Calibration.estimateYawMisalignment)
 *     compared accel angles in the phone frame with compass headings,
 *     which only works if the leveled phone frame happens to point
 *     north; its "forward" axis was close to arbitrary.
 *
 * Both gates have a hard safety cap (maxLevelSamples / maxYawSamples):
 * a genuinely noisy environment — road vibration through a mount, a
 * stop-start city start — must degrade to a best-effort estimate, never
 * to a demo stuck forever on "Calibrating…". Bounded-wait, not
 * indefinite-wait.
 */
class CalibrationManager(
    private val levelSamples: Int = 20,
    // Usable forward-axis samples (above Calibration.FORWARD_AXIS_MIN_SPEED,
    // 10 Hz) before the first estimate: the offline fit's own floor.
    private val minYawSamples: Int = Calibration.FORWARD_AXIS_MIN_SAMPLES,
    // Safety cap: past this many usable samples, lock in whatever the
    // estimate is, flagged rough (see class doc). 60 s of driving.
    private val maxYawSamples: Int = 600,
    // Wall-clock cap on the yaw stage, in ticks with or without GNSS
    // (10 Hz, so 30 s). The sample caps above only count samples above
    // 3 m/s, so a phone that never moves that fast (a walk — every
    // bundled demo replay tops out under 2 m/s — a desk, a parked car)
    // would otherwise sit at "calibrating" forever. Past this, yaw locks
    // in best-effort, flagged rough, and keeps refining (see class doc)
    // so a later real drive still fixes it.
    private val maxYawWaitTicks: Int = 300,
    // Min orientation steadiness (0-1) for a leveling window to be
    // trusted — see orientationSteadiness. A resting phone scores
    // ~0.9999; a real car driving, braking and cornering still scores
    // ~0.99 (its orientation barely changes); a phone being picked up or
    // carried drops well below.
    private val steadyOrientationMinR: Float = 0.99f,
    private val maxLevelSamples: Int = 150,
    // Two yaw estimates this far apart (radians) still count as agreeing.
    // ~8.6 degrees.
    private val yawStabilityThreshRad: Float = 0.15f,
    // Min ForwardAxisFit.coherence to lock in as converged. The bundled
    // real drive (replay_drive.json) measures 0.86-0.96 from its first
    // 100 usable samples on; pure sensor noise ~0.05.
    private val minYawCoherence: Float = 0.5f,
    // Usable samples between checkpoints (1 s at 10 Hz).
    private val yawCheckpointGap: Int = 10,
    // Consecutive checkpoints that must all agree before locking in. On
    // the bundled real drive the estimate is still 22 degrees off at 100
    // samples and within 5 of its final value from 150 on.
    private val yawRequiredAgreements: Int = 3,
    private val dt: Float = 0.1f,
    // Samples whose accel direction is further than this from the current
    // gravity estimate stay out of the drive-long gravity mean (see
    // addYawSample). A car's own acceleration tilts the measured vector by
    // under ~17 degrees even at 0.3 g; a phone picked up or re-mounted
    // turns it much further.
    private val gravityGateRad: Float = 0.52f,
) {
    private val levelBuf = ArrayList<FloatArray>(levelSamples)
    private val forwardAxis = ForwardAxisEstimator(dt)
    // Running sum of raw accel for the drive-long gravity estimate.
    private val gravitySum = DoubleArray(3)
    private var gravityCount = 0

    private var levelSamplesSeen = 0
    private var lastYawEstimate: Float? = null
    private var lastYawCheckpointSamples = 0
    private var yawAgreementStreak = 0
    private var yawWaitTicks = 0

    /** The latest forward-axis fit, null until there is one. */
    var yawFit: ForwardAxisFit? = null
        private set

    var rLevel: FloatArray? = null
        private set
    var psiYaw: Float = 0f
        private set
    var isReady: Boolean = false
        private set

    /** True if leveling settled on a genuinely stationary window; false if
     * it hit maxLevelSamples first and fell back to a best-effort estimate.
     * Same idea for yaw via [yawConverged] — surfaced so the UI can be
     * honest about a degraded estimate instead of silently pretending
     * every calibration is equally good. */
    var levelConverged: Boolean = false
        private set
    var yawConverged: Boolean = false
        private set

    val isLeveled: Boolean get() = rLevel != null

    /** True while leveling has a full window but is still rejecting it as
     * too unsteady — i.e. the user was asked to hold the phone still and
     * it genuinely isn't. Lets the UI say that specifically instead of an
     * unchanging "leveling…" that looks like a hang. */
    val levelWaitingForStillness: Boolean
        get() = !isLeveled && levelBuf.size >= levelSamples

    /** True if either stage fell back to a best-effort estimate instead of
     * converging — worth telling the user, since everything downstream
     * inherits the error. */
    val isRoughCalibration: Boolean
        get() = isReady && !(levelConverged && yawConverged)

    /** 0-100, for a progress bar — leveling's first half, yaw alignment's
     * second half. Purely a UI convenience, not used by calibration logic
     * itself. */
    val progressPercent: Int get() = when {
        isReady -> 100
        isLeveled -> 50 + maxOf(
            50 * forwardAxis.samples / minYawSamples,
            50 * yawWaitTicks / maxYawWaitTicks,
        ).coerceIn(0, 49)  // 100 only once isReady
        else -> (50 * levelSamplesSeen / levelSamples).coerceIn(0, 50)
    }

    /** Feed one raw accel sample while leveling hasn't completed yet.
     * Keeps a sliding window of the last `levelSamples` readings and only
     * commits once that window looks genuinely stationary — see class doc
     * for why "the first N samples" isn't good enough live. */
    fun addLevelSample(rawAccel: FloatArray) {
        if (isLeveled) return
        levelSamplesSeen++
        levelBuf.add(rawAccel)
        if (levelBuf.size > levelSamples) levelBuf.removeAt(0)
        if (levelBuf.size < levelSamples) return

        val stationary = orientationSteadiness(levelBuf) > steadyOrientationMinR
        // Bounded wait: commit to a best-effort estimate rather than
        // stalling the demo forever if it never settles (see class doc).
        val giveUp = levelSamplesSeen >= maxLevelSamples
        if (stationary || giveUp) {
            rLevel = Calibration.levelingRotation(Calibration.estimateGravityVector(levelBuf))
            levelConverged = stationary
            for (a in levelBuf) addGravity(a)
        }
    }

    /** Feed one raw (accel, gyro) sample with its GNSS speed, every tick
     * GNSS has a fix, once isLeveled — before and after isReady, at any
     * speed (the estimator itself only counts samples above
     * Calibration.FORWARD_AXIS_MIN_SPEED, but needs the slower ones
     * around them for dv/dt). Call [yawGnssLost] on a tick without a fix
     * so no derivative is taken across the gap. See class doc for when
     * yaw locks in and how it keeps refining after that. */
    fun addYawSample(rawAccel: FloatArray, rawGyro: FloatArray, gpsSpeed: Float) {
        val r = rLevel ?: return
        if (!gpsSpeed.isFinite()) {
            yawGnssLost()
            return
        }
        if (gravityCount == 0 || withinGravityGate(rawAccel)) addGravity(rawAccel)
        forwardAxis.add(rawAccel, rawGyro, gpsSpeed, r)
        if (countYawWait()) return

        val n = forwardAxis.samples
        if (n < minYawSamples || n - lastYawCheckpointSamples < yawCheckpointGap) return
        lastYawCheckpointSamples = n
        relevel()
        val fit = forwardAxis.fit(rLevel!!, minSamples = minYawSamples - 1) ?: return
        yawFit = fit
        val coherent = fit.coherence >= minYawCoherence

        if (isReady) {
            // Refining: follow the whole-drive estimate, and upgrade a
            // rough lock once the data supports it. Never downgrade —
            // the UI has already shown this calibration as good.
            if (coherent) {
                psiYaw = fit.psi
                yawConverged = true
            }
            return
        }

        val agreesWithLast = lastYawEstimate?.let { angleDiff(fit.psi, it) < yawStabilityThreshRad } == true
        yawAgreementStreak = if (agreesWithLast) yawAgreementStreak + 1 else 0
        lastYawEstimate = fit.psi
        val settled = yawAgreementStreak >= yawRequiredAgreements - 1 && coherent
        val giveUp = n >= maxYawSamples
        if (settled || giveUp) {
            psiYaw = fit.psi
            yawConverged = settled
            isReady = true
        }
    }

    /**
     * Leveling over the drive, as the offline calibration does it: gravity
     * is the mean of every raw accel sample so far (the steady window
     * leveling started from, then each GNSS tick), not just the first
     * two seconds. A car's acceleration averages out over a drive but not
     * over a 2 s window, and on the bundled real drive 1.5 degrees of
     * window tilt moved the forward axis by 13. Re-leveled at each yaw
     * checkpoint (once a second at most), only while GNSS is available.
     */
    private fun relevel() {
        if (gravityCount == 0) return
        val g = floatArrayOf(
            (gravitySum[0] / gravityCount).toFloat(),
            (gravitySum[1] / gravityCount).toFloat(),
            (gravitySum[2] / gravityCount).toFloat(),
        )
        val norm = sqrt((g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).toDouble())
        if (norm < 1e-6) return
        rLevel = Calibration.levelingRotation(g)
    }

    private fun addGravity(a: FloatArray) {
        if (!(a[0].isFinite() && a[1].isFinite() && a[2].isFinite())) return
        for (i in 0 until 3) gravitySum[i] = gravitySum[i] + a[i].toDouble()
        gravityCount++
    }

    private fun withinGravityGate(a: FloatArray): Boolean {
        val gx = gravitySum[0]; val gy = gravitySum[1]; val gz = gravitySum[2]
        val gn = sqrt(gx * gx + gy * gy + gz * gz)
        val an = sqrt((a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).toDouble())
        if (gn < 1e-9 || an < 1e-6) return false
        val cosAng = (gx * a[0] + gy * a[1] + gz * a[2]) / (gn * an)
        return cosAng >= kotlin.math.cos(gravityGateRad.toDouble())
    }

    /** GNSS has no fix this tick: ends the current contiguous stretch. */
    fun yawGnssLost() {
        if (!isLeveled) return
        forwardAxis.gap()
        countYawWait()
    }

    /** One more tick of the yaw stage; true if that hit maxYawWaitTicks
     * and ended it best-effort. */
    private fun countYawWait(): Boolean {
        if (isReady) return false
        yawWaitTicks++
        if (yawWaitTicks < maxYawWaitTicks) return false
        relevel()
        val fit = forwardAxis.fit(rLevel!!, minSamples = 0)
        psiYaw = if (fit != null && fit.samples >= yawCheckpointGap && fit.coherence >= minYawCoherence) fit.psi else 0f
        yawFit = fit
        yawConverged = false
        isReady = true
        return true
    }

    /**
     * Force both stages done with an identity/zero estimate — for the
     * hidden Developer Mode's "Skip calibration" testing control ONLY,
     * never on the real calibration path. Real calibration deliberately
     * waits for the phone to actually be moving above 3 m/s (see
     * class doc) — there's no way to make that genuinely fast without
     * weakening the checks this class exists for, and several of them
     * (settledness, the agreement streak) were added after real bugs a
     * faster/looser gate would have let back in. This exists so testing
     * the pipeline past calibration doesn't require physically walking or
     * driving every time.
     *
     * Marked not-converged on whichever stage(s) it actually skipped, so
     * the UI's existing "rough calibration" warning still fires — this is
     * a real shortcut, not something to quietly pass off as genuine.
     * Already-converged stages (e.g. leveling finished for real before
     * yaw got skipped) are left alone rather than overwritten.
     */
    fun skipForTesting() {
        if (rLevel == null) {
            // A flat identity matrix here means "gravity is already along
            // Z" — almost never true for a phone actually in someone's
            // hand. calibrate() would then leave the real gravity
            // component sitting in the calibrated accel, which the INS
            // integrates twice (accel -> velocity -> position) into wild,
            // fast-diverging drift the instant GPS drops — exactly the
            // scribbled, radiating trails seen live on a real device.
            // Whatever's already in levelBuf (even a handful of samples,
            // short of the full stationary window addLevelSample waits
            // for) gives a real best-effort gravity direction instead;
            // only an empty buffer falls back to identity.
            rLevel = if (levelBuf.isNotEmpty()) {
                for (a in levelBuf) addGravity(a)
                Calibration.levelingRotation(Calibration.estimateGravityVector(levelBuf))
            } else {
                floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f)
            }
            levelConverged = false
        }
        if (!isReady) {
            psiYaw = 0f
            yawConverged = false
            isReady = true
        }
    }

    /** Calibrate one raw [ax,ay,az] or [gx,gy,gz] sample — leveling-only
     * (psiYaw=0) once leveled but before yaw locks in, full calibration
     * once isReady. Returns the raw sample unchanged if not even leveled
     * yet (nothing usable to apply). */
    fun calibrate(raw: FloatArray): FloatArray {
        val r = rLevel ?: return raw
        return Calibration.calibrateSample(raw, r, psiYaw)
    }

    /**
     * How steady the phone's *orientation* was across a window: mean
     * resultant length of the normalized accel vectors, 1 = every sample
     * pointed the same way, lower = the phone was being rotated.
     *
     * Direction, not magnitude — leveling estimates a direction (which way
     * is down), so that's what has to be stable for the estimate to be
     * meaningful. Gating on |accel| stability instead looks equivalent and
     * isn't: found on a real device, where the replay demo (a real car
     * already at speed) never passed a magnitude gate, because braking and
     * cornering swing |accel| around by m/s^2 at a time. The car's
     * *orientation* is rock steady throughout, and gravity's direction in
     * the phone frame with it — so leveling was perfectly estimable the
     * whole time and the gate was rejecting good data, costing 15s and
     * flagging a fine calibration as degraded. A phone being picked up or
     * carried genuinely does swing its orientation, and still fails this.
     */
    private fun orientationSteadiness(samples: List<FloatArray>): Float {
        var sx = 0.0; var sy = 0.0; var sz = 0.0
        var counted = 0
        for (s in samples) {
            val mag = sqrt((s[0] * s[0] + s[1] * s[1] + s[2] * s[2]).toDouble())
            if (mag < 1e-6) continue  // freefall/degenerate — no direction to speak of
            sx += s[0] / mag; sy += s[1] / mag; sz += s[2] / mag
            counted++
        }
        if (counted == 0) return 0f
        val n = counted.toDouble()
        return sqrt((sx / n) * (sx / n) + (sy / n) * (sy / n) + (sz / n) * (sz / n)).toFloat()
    }

    private fun angleDiff(a: Float, b: Float): Float {
        var d = (a - b).toDouble()
        while (d > Math.PI) d -= 2 * Math.PI
        while (d < -Math.PI) d += 2 * Math.PI
        return abs(d).toFloat()
    }
}
