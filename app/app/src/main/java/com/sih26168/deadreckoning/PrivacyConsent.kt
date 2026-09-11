package com.sih26168.deadreckoning

import android.content.Context

/**
 * Whether the user has been through PrivacyConsentActivity's disclosure
 * screen and agreed to it — checked by MainActivity before it ever
 * requests the OS location permission (see PrivacyConsentActivity's own
 * doc comment). A plain persisted boolean, same simple SharedPreferences-
 * object pattern as DeviceIdentity.kt: this gate has nothing to sync or
 * revoke server-side, it's purely "has this install seen the disclosure."
 * Actually revoking location access itself is the real OS permission
 * (see privacy_point_5_body) — this flag only ever gates whether the
 * disclosure screen shows again, not whether location works.
 */
object PrivacyConsent {
    private const val PREFS_NAME = "privacy_consent_prefs"
    private const val KEY_CONSENTED = "consented"

    fun hasConsented(context: Context): Boolean =
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .getBoolean(KEY_CONSENTED, false)

    fun setConsented(context: Context) {
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putBoolean(KEY_CONSENTED, true)
            .apply()
    }
}
