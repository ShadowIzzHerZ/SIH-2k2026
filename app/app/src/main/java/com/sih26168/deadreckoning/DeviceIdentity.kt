package com.sih26168.deadreckoning

import android.content.Context
import java.util.UUID

/**
 * A stable, per-install device identifier — generated once and persisted,
 * not derived from any hardware ID (Android's own IMEI/serial APIs are
 * either unavailable to normal apps or explicitly discouraged for privacy
 * reasons; a random UUID we control is the standard replacement).
 *
 * This is the one identifier meant to be shared across both optional
 * network-facing pieces: Supabase's `devices.device_id` column
 * (see ../../SUPABASE.md) and, if/when it's wired up, the AWS backend's
 * JWT `device_id` claim. The dead-reckoning pipeline itself never touches
 * this — it has no concept of a "device" beyond raw sensor input.
 */
object DeviceIdentity {
    private const val PREFS_NAME = "device_identity_prefs"
    private const val KEY_DEVICE_ID = "device_id"

    fun get(context: Context): String {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        prefs.getString(KEY_DEVICE_ID, null)?.let { return it }

        val newId = UUID.randomUUID().toString()
        prefs.edit().putString(KEY_DEVICE_ID, newId).apply()
        return newId
    }
}
