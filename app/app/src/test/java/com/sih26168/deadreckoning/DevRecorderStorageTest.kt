package com.sih26168.deadreckoning

import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

/** Where DevRecorder puts its CSVs when external storage is or isn't there. */
class DevRecorderStorageTest {

    @get:Rule
    val tmp = TemporaryFolder()

    @Test
    fun `uses external storage when it is available`() {
        val external = tmp.newFolder("external")
        val internal = tmp.newFolder("internal")
        val dir = recordingsDir(external, internal)
        assertEquals(File(external, "recordings"), dir)
        assertTrue(dir!!.isDirectory)
    }

    @Test
    fun `falls back to internal storage when external is null`() {
        val internal = tmp.newFolder("internal")
        assertEquals(File(internal, "recordings"), recordingsDir(null, internal))
    }

    @Test
    fun `falls back when the external folder cannot be created`() {
        // A plain file where the external root should be: mkdirs fails.
        val notADir = tmp.newFile("external")
        val internal = tmp.newFolder("internal")
        assertEquals(File(internal, "recordings"), recordingsDir(notADir, internal))
    }

    @Test
    fun `returns null when no storage is usable`() {
        assertNull(recordingsDir(null, tmp.newFile("internal")))
    }
}
