package com.sih26168.deadreckoning

import kotlin.math.cos
import kotlin.math.sqrt

/**
 * Catches a GNSS Doppler speed that the GNSS positions themselves don't
 * back up. Indoors a Galaxy F17 reported a steady 8-10 km/h (2-3 m/s)
 * while lying still: above any speed floor that would still let a walk
 * through, and with the phone in a hand the IMU doesn't look still
 * either. But a phone really moving at 2.5 m/s is 12 m away 5 s later,
 * and one that is only reporting noise is not.
 *
 * Over the last [windowTicks] ticks (5 s at 10 Hz) this compares the
 * distance the reported speeds add up to with how far the fixes actually
 * went, start to end. If the path is long enough to judge ([minPathM])
 * and the fixes covered less than [minProgressRatio] of it, the speed
 * reads as 0. Only speeds under [maxCheckedSpeedMps] are ever checked, so
 * road speeds pass straight through.
 *
 * The ratio is set against data/own_recordings (real walks, 0.5-2 m/s,
 * 1 Hz fixes): there the fixes cover 85% or more of the reported path in
 * 95% of windows, and under 30% in 0.04% of them.
 */
class GnssSpeedCheck(
    private val windowTicks: Int = 50,
    private val dt: Float = 0.1f,
    private val maxCheckedSpeedMps: Float = 4.0f,
    private val minPathM: Float = 5.0f,
    private val minProgressRatio: Float = 0.3f,
) {
    private val lats = ArrayDeque<Double>(windowTicks + 1)
    private val lons = ArrayDeque<Double>(windowTicks + 1)
    private val speeds = ArrayDeque<Float>(windowTicks + 1)

    /** Call when there is no fix this tick: a window must not span a gap. */
    fun reset() { lats.clear(); lons.clear(); speeds.clear() }

    /** [speedMps] as reported for the fix at [lat]/[lon] this tick, or 0
     * when the last [windowTicks] ticks of fixes say it isn't real. */
    fun filter(lat: Double, lon: Double, speedMps: Float): Float {
        if (lats.size > windowTicks) { lats.removeFirst(); lons.removeFirst(); speeds.removeFirst() }
        lats.addLast(lat); lons.addLast(lon); speeds.addLast(speedMps)
        if (speedMps <= 0f || speedMps >= maxCheckedSpeedMps) return speedMps
        if (lats.size <= windowTicks) return speedMps
        // speeds[0] belongs to the interval before the window's first fix.
        val pathM = (speeds.sum() - speeds.first()) * dt
        if (pathM < minPathM) return speedMps
        val r = 6371000.0
        val dx = Math.toRadians(lons.last() - lons.first()) * r * cos(Math.toRadians(lats.first()))
        val dy = Math.toRadians(lats.last() - lats.first()) * r
        val netM = sqrt(dx * dx + dy * dy)
        return if (netM < minProgressRatio * pathM) 0f else speedMps
    }
}
