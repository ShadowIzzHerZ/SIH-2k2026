package com.sih26168.deadreckoning

import kotlin.math.sqrt

/**
 * On-device self-check for real drives: logs which model answered each
 * 5 s chunk and how far the dead-reckoned position drifted during each
 * blackout, so `adb logcat -s DriveSelfCheck` after a drive says what
 * happened without a debugger. Plain Kotlin (the log sink is a lambda) so
 * DriveSelfCheckTest drives it on the JVM.
 *
 * Drift is reported the way the offline evaluation reports it: position
 * error over distance travelled, in percent.
 *  - During a demo blackout (or a replay), GNSS is really still there, so
 *    the error against it is known live, every second.
 *  - During a real blackout there is no truth until GNSS returns; the end
 *    error is the dead-reckoned position at the last blackout tick against
 *    the first fix after it, over the dead-reckoned path length.
 */
class DriveSelfCheck(
    private val log: (String) -> Unit,
    private val dt: Float = 0.1f,
    private val liveEverySeconds: Float = 1f,
) {
    /** One finished blackout. truth* distances come from GNSS (demo
     * blackouts, replays); otherwise from the dead-reckoned path. */
    data class BlackoutReport(
        val seconds: Float,
        val distanceM: Float,
        val endErrorM: Float?,
        val driftPercent: Float?,
        val truthDuringBlackout: Boolean,
        val slowChunks: Int,
        val fastChunks: Int,
        val zeroChunks: Int,
        val fallbacks: Int,
    )

    val reports = ArrayList<BlackoutReport>()

    private var inBlackout = false
    private var ticks = 0
    private var drDistance = 0.0
    private var truthDistance = 0.0
    private var truthAllAlong = true
    private var lastDr: FloatArray? = null
    private var lastTruth: FloatArray? = null
    private var pendingEnd: FloatArray? = null  // DR at the last blackout tick, awaiting a fix
    private var pendingTicks = 0
    private var chunksAtStart = IntArray(4)
    private var lastChunks = 0
    private var model: EnsembleModel? = null
    private var ticksSinceLive = 0

    /**
     * One tick after FusionEngine.tick. [dr] is the fused state's (x, y),
     * [truth] the real GNSS fix in the same local frame when there is one
     * (also during a demo blackout), null otherwise. [ensemble] is the
     * running model, for which-model-answered logging.
     */
    fun tick(mode: FusionMode, dr: FloatArray, truth: FloatArray?, ensemble: EnsembleModel?) {
        model = ensemble
        logNewChunk(ensemble)

        // A real blackout that ended: score it against the first fix.
        pendingEnd?.let { end ->
            if (truth != null) {
                finish(end, truth)
            } else if (++pendingTicks >= MAX_WAIT_FOR_FIX_TICKS) {
                finish(end, null)
            }
        }

        val blackout = mode == FusionMode.BLACKOUT
        if (blackout && !inBlackout) start(ensemble)
        if (blackout) {
            ticks++
            lastDr?.let { drDistance += dist(it, dr) }
            if (truth != null && lastTruth != null) truthDistance += dist(lastTruth!!, truth)
            if (truth == null) truthAllAlong = false
            if (truth != null && truthAllAlong && ++ticksSinceLive * dt >= liveEverySeconds) {
                ticksSinceLive = 0
                val err = dist(dr, truth)
                log(
                    "blackout %.1f s: %.0f m travelled, error %.1f m (%s) [live, GNSS kept running]".format(
                        ticks * dt, truthDistance, err, pct(err, truthDistance),
                    )
                )
            }
        } else if (inBlackout) {
            inBlackout = false
            val sameTimeTruth = lastTruth
            if (truthAllAlong && sameTimeTruth != null) {
                // Demo blackout: score against GNSS at the same tick.
                finish(lastDr!!, sameTimeTruth)
            } else if (truth != null) {
                finish(lastDr!!, truth)
            } else {
                pendingEnd = lastDr
                pendingTicks = 0
            }
        }
        lastDr = dr.copyOf()
        lastTruth = truth?.copyOf()
    }

    private fun start(ensemble: EnsembleModel?) {
        pendingEnd?.let { finish(it, null) }  // the previous one never got a fix
        inBlackout = true
        ticks = 0
        drDistance = 0.0
        truthDistance = 0.0
        truthAllAlong = true
        ticksSinceLive = 0
        // Distances count only between blackout ticks.
        lastDr = null
        lastTruth = null
        chunksAtStart = counts(ensemble)
        log("blackout started")
    }

    private fun finish(drEnd: FloatArray, truth: FloatArray?) {
        pendingEnd = null
        val c = counts(model)
        val useTruth = truthAllAlong && truthDistance > 0
        val distance = (if (useTruth) truthDistance else drDistance).toFloat()
        val err = truth?.let { dist(drEnd, it).toFloat() }
        val report = BlackoutReport(
            seconds = ticks * dt,
            distanceM = distance,
            endErrorM = err,
            driftPercent = err?.let { if (distance > 1f) 100f * it / distance else null },
            truthDuringBlackout = useTruth,
            slowChunks = c[0] - chunksAtStart[0],
            fastChunks = c[1] - chunksAtStart[1],
            zeroChunks = c[2] - chunksAtStart[2],
            fallbacks = c[3] - chunksAtStart[3],
        )
        reports += report
        log(
            ("blackout ended after %.1f s: %.0f m (%s), end error %s, drift %s; " +
                "chunks slow %d / fast %d / zero %d, fallbacks %d").format(
                report.seconds, report.distanceM, if (useTruth) "GNSS path" else "dead-reckoned path",
                err?.let { "%.1f m".format(it) } ?: "unknown (no fix after it)",
                report.driftPercent?.let { "%.1f%%".format(it) } ?: "n/a",
                report.slowChunks, report.fastChunks, report.zeroChunks, report.fallbacks,
            )
        )
    }

    private fun logNewChunk(ensemble: EnsembleModel?) {
        val e = ensemble ?: return
        val n = e.chunks
        if (n == lastChunks) return
        lastChunks = n
        val who = when (e.lastUsed) {
            ModelRole.FAST -> "fast model"
            ModelRole.SLOW -> "slow model"
            null -> "no model (zero corrections)"
        }
        log("chunk %d: %s, start speed %.1f m/s%s".format(n, who, e.lastStartSpeed, if (e.fallbackCount > 0) ", fallbacks so far ${e.fallbackCount}" else ""))
    }

    private fun counts(e: EnsembleModel?) =
        if (e == null) IntArray(4) else intArrayOf(e.slowChunks, e.fastChunks, e.zeroChunks, e.fallbackCount)

    private fun pct(err: Double, distance: Double) = if (distance > 1.0) "%.1f%%".format(100 * err / distance) else "n/a"

    private fun dist(a: FloatArray, b: FloatArray): Double {
        val dx = (a[0] - b[0]).toDouble(); val dy = (a[1] - b[1]).toDouble()
        return sqrt(dx * dx + dy * dy)
    }

    companion object {
        /** After a real blackout, how long to wait for a fix to score it
         * against before reporting the end error as unknown. 30 s. */
        const val MAX_WAIT_FOR_FIX_TICKS = 300
    }
}
