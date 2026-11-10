package com.sih26168.deadreckoning

import android.annotation.SuppressLint
import android.content.Context
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Bundle
import android.os.SystemClock

/**
 * Wraps LocationManager directly — no Play Services / API key dependency,
 * simplest reliable option for a hackathon demo. Exposes the most recent
 * fix plus how long ago it arrived, so the tick loop can decide "GNSS
 * available" the same way a real deployment would: no fix recently enough,
 * not "no fix ever".
 *
 * Location.getSpeed()/getBearing() are the platform's own Doppler-derived
 * values, not position-differenced by this app — exactly the GNSS-chip
 * preference src/fusion.py's run_fusion docstring establishes (confirmed
 * in testing: differencing two low-speed position fixes produced a 103
 * m/s spike from GPS noise alone).
 */
class LocationReader(private val context: Context) : LocationListener {
    private val locationManager = context.getSystemService(Context.LOCATION_SERVICE) as LocationManager

    @Volatile var lastLocation: Location? = null
        private set
    @Volatile var lastFixUptimeMs: Long = 0
        private set
    // Newest live fix with a Doppler speed (in practice: from GPS), the
    // only one fusion and yaw calibration read. Kept apart from
    // lastLocation because a NETWORK_PROVIDER fix (no speed, no bearing)
    // landing there used to replace a good GPS fix: hasRecentFix() went
    // false, fusion dropped to BLACKOUT, and the next GPS fix brought it
    // back through BLEND ("GPS reconnecting"), every time the network
    // provider delivered, while GPS itself never stopped. Only set from
    // onLocationChanged, so start()'s cached getLastKnownLocation() seed
    // (minutes or hours old) places the marker but never reaches fusion.
    @Volatile var lastFusionFix: Location? = null
        private set
    @Volatile private var lastFusionFixUptimeMs: Long = 0

    @SuppressLint("MissingPermission")  // caller (MainActivity) checks permission before start()
    fun start() {
        // Both providers, not "GPS if enabled, else network" — a real bug
        // found live: GPS_PROVIDER reporting enabled=true doesn't mean it
        // has ever produced a fix. `adb shell dumpsys location` on an
        // actual indoor test showed the raw GNSS provider sitting at
        // `last location=null` (cold satellite lock can take a long time
        // indoors, sometimes never completes) while NETWORK_PROVIDER
        // already had a perfectly usable Wi-Fi/cell-based fix ready
        // immediately — but the old single-provider choice meant that
        // fix was never requested or read, so the map showed no "you are
        // here" marker at all for the entire time GPS hadn't locked yet,
        // including the whole calibration phase. Requesting updates from
        // every enabled provider means whichever produces a fix first
        // (almost always network, initially) drives the marker, and GPS
        // naturally takes over once it locks (see onLocationChanged —
        // it just accepts whatever arrives, from either provider).
        //
        // Registered even while a provider is switched off (location off in
        // quick settings): LocationManager then starts delivering as soon
        // as the user turns it on. Skipping disabled providers here meant
        // turning location on after launch never produced a fix until the
        // app was restarted. Only providers the device doesn't have at all
        // are left out (requesting those throws).
        val providers = locationProvidersToRequest(locationManager.allProviders)
        if (providers.isEmpty()) return
        for (provider in providers) {
            locationManager.requestLocationUpdates(provider, /* minTimeMs= */ 500L, /* minDistanceM= */ 0f, this)
        }
        // Seed immediately from whichever provider already has a last-known
        // fix — take the more recent of the two if both exist, so a fresh
        // GPS fix isn't clobbered by a stale cached network one.
        providers.mapNotNull { locationManager.getLastKnownLocation(it) }
            .maxByOrNull { it.time }
            ?.let {
                lastLocation = it
                lastFixUptimeMs = SystemClock.elapsedRealtime()
            }
    }

    fun stop() {
        locationManager.removeUpdates(this)
    }

    /** No fix within this window counts as a blackout, even without the
     * manual demo toggle — a real device losing satellite lock (tunnel,
     * underground parking) stops delivering updates rather than reporting
     * "unavailable" explicitly. Reads lastFusionFix, not lastLocation, so a
     * speedless network fix can't knock out a live GPS one.
     *
     * Needs hasSpeed() but no longer hasBearing(): a stationary GPS chip
     * reports a valid fix with zero speed and no course, and requiring a
     * bearing turned every stop into a fake blackout (speed then came from
     * dead reckoning and drifted). FusionEngine.tick() already keeps its
     * previous heading when gnssHasBearing is false, and yaw calibration
     * checks hasBearing itself.
     *
     * 3 s instead of 2 s: GPS indoors or under cover arrives at ~1 Hz with
     * the odd 2 s gap, which on its own flipped tracking off and on. */
    fun hasRecentFix(maxAgeMs: Long = 3000L): Boolean {
        val loc = lastFusionFix ?: return false
        return SystemClock.elapsedRealtime() - lastFusionFixUptimeMs < maxAgeMs && loc.hasSpeed()
    }

    /** Weaker than hasRecentFix() — just "do we know roughly where the
     * phone is", for showing the "you are here" marker. Two real bugs
     * found live, both from the same over-tight check: (1) while
     * genuinely stationary (e.g. the leveling calibration step, which
     * explicitly tells the user to hold the phone still), a lot of
     * Android GPS chips report a valid lat/lon fix but leave hasBearing()
     * false (no meaningful course of travel at zero velocity) — so
     * hasRecentFix() correctly refuses to treat that as "available for
     * fusion", but gating the marker on the same check meant no dot for
     * the entire leveling phase despite the phone's location already
     * being known; (2) the default 2000ms staleness window, copied from
     * hasRecentFix(), assumes dense live-GPS-rate updates — but
     * confirmed live via `adb shell dumpsys location`, NETWORK_PROVIDER
     * (see start()'s own doc for why the marker needs this provider at
     * all) only actually delivers a fix roughly every ~20s, throttled by
     * the platform's network location backend regardless of the 500ms
     * minTime requested. A 2000ms window against a ~20s delivery cadence
     * meant the marker was only ever "recent enough" for ~2 of every 20
     * seconds — it kept appearing then vanishing, reading as "the dot
     * isn't there" for almost the entire time anyone was actually
     * looking at the screen. 30s comfortably covers that real cadence. */
    fun hasAnyRecentFix(maxAgeMs: Long = 30_000L): Boolean {
        val loc = lastLocation ?: return false
        return SystemClock.elapsedRealtime() - lastFixUptimeMs < maxAgeMs
    }

    override fun onLocationChanged(location: Location) {
        val now = SystemClock.elapsedRealtime()
        lastLocation = location
        lastFixUptimeMs = now
        if (location.hasSpeed()) {
            lastFusionFix = location
            lastFusionFixUptimeMs = now
        }
    }

    @Deprecated("Deprecated in Java", ReplaceWith(""))
    override fun onStatusChanged(provider: String?, status: Int, extras: Bundle?) {}
    override fun onProviderEnabled(provider: String) {}
    override fun onProviderDisabled(provider: String) {}
}

/** GPS and network, in that order, minus any this device doesn't have
 * ([allProviders] is LocationManager.getAllProviders()). Enabled or not
 * doesn't matter, see LocationReader.start(). */
internal fun locationProvidersToRequest(allProviders: List<String>): List<String> =
    listOf(LocationManager.GPS_PROVIDER, LocationManager.NETWORK_PROVIDER).filter { it in allProviders }

/** GNSS Doppler speed with the stationary noise floor removed. A phone
 * lying still typically still reports a few tenths of a m/s, more than
 * 1 m/s indoors with multipath, and fusion shows that speed as-is, so
 * the speed tile jittered while nobody moved. Anything under [floorMps], or under one
 * reported standard deviation ([speedAccuracyMps], NaN when the chip
 * gives none) while still slow, reads as 0. Real motion above
 * [maxNoiseMps] always passes through untouched. */
internal fun stationaryFilteredSpeed(
    speedMps: Float,
    speedAccuracyMps: Float,
    floorMps: Float = 0.5f,
    maxNoiseMps: Float = 2.0f,
): Float {
    if (speedMps < floorMps) return 0f
    if (speedMps < maxNoiseMps && !speedAccuracyMps.isNaN() && speedMps < speedAccuracyMps) return 0f
    return speedMps
}
