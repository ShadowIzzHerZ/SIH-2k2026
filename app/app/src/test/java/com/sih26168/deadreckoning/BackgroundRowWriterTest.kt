package com.sih26168.deadreckoning

import java.io.IOException
import java.io.StringWriter
import java.io.Writer
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** DevRecorder's writer: rows go to the file off the calling thread, in
 * order, and are all on disk once close() returns. */
class BackgroundRowWriterTest {

    /** Records which threads wrote to it and whether it was closed. */
    private class RecordingWriter(private val failOnWrite: Int = -1) : Writer() {
        val text = StringWriter()
        val threads = mutableSetOf<String>()
        var writes = 0
        var flushed = false
        var closed = false
        override fun write(cbuf: CharArray, off: Int, len: Int) {
            threads.add(Thread.currentThread().name)
            if (writes++ == failOnWrite) throw IOException("disk full")
            text.write(cbuf, off, len)
        }
        override fun flush() { flushed = true }
        override fun close() { closed = true }
    }

    @Test
    fun `rows are written in order off the calling thread and closed`() {
        val out = RecordingWriter()
        val w = BackgroundRowWriter(out)
        val rows = (0 until 1000).map { "row $it\n" }
        rows.forEach(w::write)
        w.close()
        assertEquals(rows.joinToString(""), out.text.toString())
        assertEquals(setOf("dev-recorder-writer"), out.threads)
        assertTrue(Thread.currentThread().name !in out.threads)
        assertTrue(out.flushed)
        assertTrue(out.closed)
        assertNull(w.failure)
    }

    @Test
    fun `a failed write is kept, later rows are dropped and close still finishes`() {
        val out = RecordingWriter(failOnWrite = 2)
        val w = BackgroundRowWriter(out)
        (0 until 5).forEach { w.write("row $it\n") }
        w.close()
        assertEquals("row 0\nrow 1\n", out.text.toString())
        assertEquals("disk full", w.failure?.message)
        assertTrue(out.closed)
    }
}
