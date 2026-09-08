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

    @SuppressLint("MissingPermission")  // caller (MainActivity) checks permission before start()
    fun start() {
        val provider = when {
            locationManager.isProviderEnabled(LocationManager.GPS_PROVIDER) -> LocationManager.GPS_PROVIDER
            locationManager.isProviderEnabled(LocationManager.NETWORK_PROVIDER) -> LocationManager.NETWORK_PROVIDER
            else -> return
        }
        locationManager.requestLocationUpdates(provider, /* minTimeMs= */ 500L, /* minDistanceM= */ 0f, this)
        locationManager.getLastKnownLocation(provider)?.let {
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
     * "unavailable" explicitly. */
    fun hasRecentFix(maxAgeMs: Long = 2000L): Boolean {
        val loc = lastLocation ?: return false
        return SystemClock.elapsedRealtime() - lastFixUptimeMs < maxAgeMs && loc.hasSpeed() && loc.hasBearing()
    }

    override fun onLocationChanged(location: Location) {
        lastLocation = location
        lastFixUptimeMs = SystemClock.elapsedRealtime()
    }

    @Deprecated("Deprecated in Java", ReplaceWith(""))
    override fun onStatusChanged(provider: String?, status: Int, extras: Bundle?) {}
    override fun onProviderEnabled(provider: String) {}
    override fun onProviderDisabled(provider: String) {}
}
