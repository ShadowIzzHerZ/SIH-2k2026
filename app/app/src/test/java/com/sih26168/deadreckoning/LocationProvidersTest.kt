package com.sih26168.deadreckoning

import org.junit.Assert.assertEquals
import org.junit.Test

/** Which providers LocationReader registers with. */
class LocationProvidersTest {

    @Test
    fun `requests gps and network whenever the device has them`() {
        assertEquals(listOf("gps", "network"), locationProvidersToRequest(listOf("passive", "network", "fused", "gps")))
    }

    @Test
    fun `skips providers the device does not have`() {
        assertEquals(listOf("network"), locationProvidersToRequest(listOf("passive", "network")))
        assertEquals(emptyList<String>(), locationProvidersToRequest(listOf("passive")))
    }
}

/** The stationary speed floor LocationReader applies before fusion. */
class StationarySpeedTest {

    @Test
    fun `speed under the floor reads as zero`() {
        assertEquals(0f, stationaryFilteredSpeed(0.4f, Float.NaN), 0f)
        assertEquals(0f, stationaryFilteredSpeed(0f, 0.1f), 0f)
    }

    @Test
    fun `slow speed inside its own accuracy reads as zero`() {
        assertEquals(0f, stationaryFilteredSpeed(1.2f, 1.5f), 0f)
    }

    @Test
    fun `slow speed outside its accuracy passes`() {
        assertEquals(1.2f, stationaryFilteredSpeed(1.2f, 0.3f), 0f)
        assertEquals(1.2f, stationaryFilteredSpeed(1.2f, Float.NaN), 0f)
    }

    @Test
    fun `real driving speed always passes, even with poor accuracy`() {
        assertEquals(8f, stationaryFilteredSpeed(8f, 10f), 0f)
    }
}
