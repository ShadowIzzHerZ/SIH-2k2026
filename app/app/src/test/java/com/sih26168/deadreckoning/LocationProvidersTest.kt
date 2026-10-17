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
