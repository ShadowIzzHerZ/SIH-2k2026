package com.sih26168.deadreckoning

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class RecordingForegroundTest {

    @Test
    fun `location foreground service needs a location permission`() {
        assertFalse(RecordingForeground.canStart(hasFineLocation = false, hasCoarseLocation = false))
        assertTrue(RecordingForeground.canStart(hasFineLocation = true, hasCoarseLocation = false))
        assertTrue(RecordingForeground.canStart(hasFineLocation = false, hasCoarseLocation = true))
        assertTrue(RecordingForeground.canStart(hasFineLocation = true, hasCoarseLocation = true))
    }

    @Test
    fun `notification permission is only requested on API 33 and up when missing`() {
        assertFalse(RecordingForeground.shouldRequestNotificationPermission(sdkInt = 26, granted = false))
        assertFalse(RecordingForeground.shouldRequestNotificationPermission(sdkInt = 32, granted = false))
        assertTrue(RecordingForeground.shouldRequestNotificationPermission(sdkInt = 33, granted = false))
        assertTrue(RecordingForeground.shouldRequestNotificationPermission(sdkInt = 34, granted = false))
        assertFalse(RecordingForeground.shouldRequestNotificationPermission(sdkInt = 34, granted = true))
    }
}
