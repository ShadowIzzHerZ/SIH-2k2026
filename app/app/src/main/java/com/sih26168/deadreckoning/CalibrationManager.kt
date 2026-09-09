package com.sih26168.deadreckoning

import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Drives Calibration.kt's two-stage leveling + yaw estimate live, on
 * startup, instead of over an already-recorded file — same two stages as
 * calibration.py's module docstring: leveling first (assumes the phone is
 * roughly still for the first `levelSamples` readings, e.g. mounted before
 * pulling off), then yaw alignment once during an initial confident
 * (moving, GPS-available) stretch, locked in after that — "leveling
 * happens continuously, yaw-alignment happens once at trip start", per
 * that module's own reasoning, which applies just as much live as offline.
 *
 * minConfidentSamples is far smaller than calibrate_sequence's offline
 * default (3000, meant for a whole recorded file) — a live app can't make
 * the user drive confidently for 5 minutes before showing anything; ~3s of
 * real motion is enough to lock in a reasonable estimate for a demo.
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
 *     if the user is still fumbling with it. Steadiness of direction
 *     rather than of |accel|, so a phone already mounted in a moving
 *     vehicle levels immediately instead of waiting out the timeout.
 *   - Yaw lock-in waits for the estimate to *settle* — two checkpoints
 *     ~1s apart agreeing — instead of locking in on whatever
 *     `minConfidentSamples` happens to produce first. The estimate
 *     assumes an initial straight-line stretch (see
 *     estimate_yaw_misalignment); a stretch containing a turn, a lane
 *     change, or stop-and-go traffic makes the raw median swing wildly
 *     as more samples arrive, and settledness catches that directly.
 *
 *     Tried circular variance of the underlying angle diffs first (do
 *     the *samples* agree with each other, not just the estimate) — it's
 *     the theoretically cleaner check, and synthetic tests confirmed it
 *     rejects an alternating-heading stretch correctly. But measuring it
 *     against this app's own bundled real recorded drive (see
 *     ReplayDataSource) found real accel-vs-GPS-heading noise is louder
 *     than that synthetic test assumed: circular variance swung from
 *     0.14 to 0.84 across different windows of the *same* validated
 *     drive, with no threshold separating "real noisy driving" from "an
 *     actual turn" — so gating on it made every real calibration run out
 *     the safety cap and get flagged degraded, on data that was fine.
 *     Checkpoint settledness alone turned out to already reject the
 *     synthetic alternating-heading case just as well (each checkpoint's
 *     median swings between the two headings' respective values, which
 *     is exactly what settledness checks for) — so it's the one gate
 *     doing real work; circular variance is kept as a much looser,
 *     informational-only signal (see [yawSpreadThresh]'s doc).
 *
 * Both gates have a hard safety cap (maxLevelSamples / maxConfidentSamples):
 * a genuinely noisy environment — road vibration through a mount, a
 * stop-start city start — must degrade to a best-effort estimate, never
 * to a demo stuck forever on "Calibrating…". Bounded-wait, not
 * indefinite-wait.
 */
class CalibrationManager(
    private val levelSamples: Int = 20,
    private val minConfidentSamples: Int = 30,
    private val maxConfidentSamples: Int = 90,
    private val minSpeedMps: Float = 2.0f,
    // Min orientation steadiness (0-1) for a leveling window to be
    // trusted — see orientationSteadiness. A resting phone scores
    // ~0.9999; a real car driving, braking and cornering still scores
    // ~0.99 (its orientation barely changes); a phone being picked up or
    // carried drops well below.
    private val steadyOrientationMinR: Float = 0.99f,
    private val maxLevelSamples: Int = 150,
    // Two yaw estimates this far apart (radians) still count as agreeing.
    // ~8.6 degrees — comfortably inside the accuracy this estimate has
    // anyway, while still rejecting a mid-turn estimate that swings tens
    // of degrees between checkpoints.
    private val yawStabilityThreshRad: Float = 0.15f,
    // Informational only (see class doc) — not part of the isReady gate.
    // Only flags isRoughCalibration for near-total incoherence (real
    // validated driving data measured up to 0.84; this sits well above
    // that so it doesn't cry wolf on real noisy-but-fine driving).
    private val yawSpreadThresh: Float = 0.95f,
    private val yawCheckpointGap: Int = 10,
    // Consecutive checkpoints (each yawCheckpointGap raw ticks apart) that
    // must all agree before locking in — not just the latest two. A
    // median over a small filtered subset can land close to a previous
    // checkpoint's value by pure chance even from genuinely inconsistent
    // data (measured directly: a synthetic 50/50 random-mix stretch with
    // no real consistent direction still passed a 2-checkpoint agreement
    // check on a single draw). Three independent agreements in a row is
    // far less likely to be coincidence.
    private val yawRequiredAgreements: Int = 3,
    // Sliding window, not an ever-growing buffer: an early turn must not
    // permanently contaminate every later estimate for the rest of the
    // session. Caught in review — a scenario with a turn in the first 20
    // samples followed by 60 samples of a real straight stretch still
    // locked in on the pre-turn offset, because the original samples
    // never left the buffer. ~2x minConfidentSamples by default: enough
    // history for a stable estimate, short enough to actually recover
    // after a turn within one confident-collection window.
    private val yawWindowSamples: Int = 60,
) {
    private val levelBuf = ArrayList<FloatArray>(levelSamples)
    private val yawAccelBuf = ArrayList<FloatArray>()
    private val yawHeadingBuf = ArrayList<FloatArray>()

    private var levelSamplesSeen = 0
    private var yawSamplesSeen = 0
    private var lastYawEstimate: Float? = null
    private var lastYawCheckpointSize = 0
    private var yawAgreementStreak = 0

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
        isLeveled -> 50 + (50 * yawSamplesSeen / minConfidentSamples).coerceIn(0, 50)
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
        }
    }

    /** Feed one (leveled accel, GPS speed, GPS heading) sample while yaw
     * hasn't locked in yet — only call once isLeveled and GNSS is
     * available with speed > minSpeedMps. Locks in once two estimates
     * `yawCheckpointGap` samples apart agree, or at maxConfidentSamples,
     * whichever comes first — see class doc. */
    fun addYawSample(rawAccel: FloatArray, gpsSpeed: Float, gpsHeadingRad: Float) {
        if (!isLeveled || isReady || gpsSpeed <= minSpeedMps) return
        val leveled = Calibration.matVec3(rLevel!!, rawAccel)
        yawAccelBuf.add(floatArrayOf(leveled[0], leveled[1]))
        yawHeadingBuf.add(
            floatArrayOf(
                kotlin.math.cos(gpsHeadingRad.toDouble()).toFloat(),
                kotlin.math.sin(gpsHeadingRad.toDouble()).toFloat(),
            )
        )
        if (yawAccelBuf.size > yawWindowSamples) {
            yawAccelBuf.removeAt(0)
            yawHeadingBuf.removeAt(0)
        }

        yawSamplesSeen++
        val n = yawSamplesSeen
        // n counts total samples seen (for the minConfidentSamples/
        // maxConfidentSamples/checkpoint-gap bookkeeping below), decoupled
        // from yawAccelBuf.size (the windowed buffer actually fed to the
        // estimator) now that the buffer can shrink.
        if (n < minConfidentSamples) return
        if (n - lastYawCheckpointSize < yawCheckpointGap && lastYawEstimate != null) return

        val estimate = Calibration.estimateYawMisalignment(yawAccelBuf, yawHeadingBuf)
        // The load-bearing check: has the estimate stopped moving across
        // yawRequiredAgreements checkpoints in a row? See class doc for
        // why circular variance didn't add anything real data didn't
        // already fail cleanly, and this streak's own doc for why two
        // checkpoints agreeing isn't enough on its own.
        val agreesWithLast = lastYawEstimate?.let { angleDiff(estimate, it) < yawStabilityThreshRad } == true
        yawAgreementStreak = if (agreesWithLast) yawAgreementStreak + 1 else 0
        val settled = yawAgreementStreak >= yawRequiredAgreements - 1
        val giveUp = n >= maxConfidentSamples

        if (settled || giveUp) {
            psiYaw = estimate
            // Near-total incoherence even in a settled estimate is still
            // worth flagging — see yawSpreadThresh's doc.
            val incoherent = Calibration.yawMisalignmentSpread(yawAccelBuf, yawHeadingBuf) >= yawSpreadThresh
            yawConverged = settled && !incoherent
            isReady = true
            return
        }
        lastYawEstimate = estimate
        lastYawCheckpointSize = n
    }

    /**
     * Force both stages done with an identity/zero estimate — for the
     * hidden Developer Mode's "Skip calibration" testing control ONLY,
     * never on the real calibration path. Real calibration deliberately
     * waits for the phone to actually be moving above minSpeedMps (see
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
            rLevel = floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f)
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
