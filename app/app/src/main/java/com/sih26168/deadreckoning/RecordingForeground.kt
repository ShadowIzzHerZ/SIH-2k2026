package com.sih26168.deadreckoning

/**
 * The two Android-version rules RecordingService has to respect, kept
 * free of Android types so they're plain-JVM testable
 * (RecordingForegroundTest).
 */
object RecordingForeground {

    /** Android 14+ (this app targets 34) throws SecurityException from
     * startForeground() with type "location" unless a location permission
     * is already granted — and the recording can be started without one
     * (replay off after "continue in demo mode"). Either fine or coarse is
     * enough for the platform's check. */
    fun canStart(hasFineLocation: Boolean, hasCoarseLocation: Boolean): Boolean =
        hasFineLocation || hasCoarseLocation

    /** POST_NOTIFICATIONS is a runtime permission from Android 13 (API 33)
     * on; below that it doesn't exist and is implicitly granted. Without
     * it the service still runs, its notification just isn't shown in the
     * drawer — so this only decides whether to ask, never whether to record. */
    fun shouldRequestNotificationPermission(sdkInt: Int, granted: Boolean): Boolean =
        sdkInt >= 33 && !granted
}
