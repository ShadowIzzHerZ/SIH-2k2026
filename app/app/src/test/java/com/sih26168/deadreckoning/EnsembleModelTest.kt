package com.sih26168.deadreckoning

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertSame
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The speed gate, the fast model's speed channel, and what the ensemble
 * does when a model is missing or misbehaves. Fakes stand in for the ONNX
 * models; the real files are checked by src/export_ensemble.py.
 */
class EnsembleModelTest {

    /** Answers every sample with [answer] and records the last input. */
    private class Recording(private val answer: FloatArray, private val channels: Int) : Predictor, AutoCloseable {
        var calls = 0
        var lastInput: Array<FloatArray>? = null
        var closed = false
        override fun predict(window: Array<FloatArray>): Array<FloatArray> {
            calls++
            lastInput = window
            require(window.all { it.size == channels }) { "wrong channel count" }
            return Array(window.size) { answer.copyOf() }
        }
        override fun close() { closed = true }
    }

    private class Throwing : Predictor {
        override fun predict(window: Array<FloatArray>): Array<FloatArray> = throw IllegalStateException("boom")
    }

    private class Returning(private val out: (Int) -> Array<FloatArray>) : Predictor {
        override fun predict(window: Array<FloatArray>) = out(window.size)
    }

    private val slowAnswer = floatArrayOf(1f, 0.1f)
    private val fastAnswer = floatArrayOf(2f, 0.2f)
    private fun slow() = Recording(slowAnswer, 6)
    private fun fast() = Recording(fastAnswer, 7)

    private fun window(n: Int = 50) = Array(n) { i -> FloatArray(6) { c -> i + c / 10f } }

    @Test
    fun `gate is 6 m per s inclusive, as in training`() {
        assertEquals(6f, SpeedGate.GATE_MPS)
        assertFalse(SpeedGate.useFast(0f))
        assertFalse(SpeedGate.useFast(5.999f))
        assertTrue(SpeedGate.useFast(6f))
        assertTrue(SpeedGate.useFast(30f))
        assertFalse(SpeedGate.useFast(-1f))
    }

    @Test
    fun `a broken speed estimate goes to the slow model`() {
        assertFalse(SpeedGate.useFast(Float.NaN))
        assertFalse(SpeedGate.useFast(Float.POSITIVE_INFINITY))
        assertFalse(SpeedGate.useFast(Float.NEGATIVE_INFINITY))
    }

    @Test
    fun `speed channel is start speed over 10 on every row, IMU channels untouched`() {
        val w = window()
        val out = SpeedGate.withSpeedChannel(w, 12.5f)
        assertEquals(50, out.size)
        for (i in w.indices) {
            assertEquals(7, out[i].size)
            assertArrayEquals(w[i], out[i].copyOfRange(0, 6), 0f)
            assertEquals(1.25f, out[i][6], 1e-6f)
        }
        // The caller's window is not modified.
        assertTrue(w.all { it.size == 6 })
        assertEquals(0f, SpeedGate.withSpeedChannel(w, Float.NaN)[0][6])
    }

    @Test
    fun `speed channel rejects a window that is not 6 channels`() {
        assertThrows(IllegalArgumentException::class.java) {
            SpeedGate.withSpeedChannel(Array(50) { FloatArray(7) }, 10f)
        }
    }

    @Test
    fun `below the gate the slow model runs on the plain 6 channels`() {
        val s = slow(); val f = fast()
        val e = EnsembleModel(s, f)
        val out = e.predict(window(), 3f)
        assertArrayEquals(slowAnswer, out[0], 0f)
        assertEquals(1, s.calls); assertEquals(0, f.calls)
        assertEquals(ModelRole.SLOW, e.lastUsed)
        assertEquals(0, e.fallbackCount)
    }

    @Test
    fun `at or above the gate the fast model runs with the speed channel`() {
        val s = slow(); val f = fast()
        val e = EnsembleModel(s, f)
        val out = e.predict(window(), 20f)
        assertArrayEquals(fastAnswer, out[0], 0f)
        assertEquals(0, s.calls); assertEquals(1, f.calls)
        assertEquals(2f, f.lastInput!![10][6], 1e-6f)
        assertEquals(ModelRole.FAST, e.lastUsed)
    }

    @Test
    fun `chunks switch models as the start speed crosses the gate`() {
        val s = slow(); val f = fast()
        val e = EnsembleModel(s, f)
        val used = listOf(2f, 5.9f, 6f, 15f, 5f, 0f).map { e.predict(window(), it); e.lastUsed }
        assertEquals(
            listOf(ModelRole.SLOW, ModelRole.SLOW, ModelRole.FAST, ModelRole.FAST, ModelRole.SLOW, ModelRole.SLOW),
            used,
        )
    }

    @Test
    fun `predict without a speed uses the slow model`() {
        val s = slow(); val f = fast()
        EnsembleModel(s, f).predict(window())
        assertEquals(1, s.calls); assertEquals(0, f.calls)
    }

    @Test
    fun `missing fast model means slow covers every speed`() {
        val s = slow()
        val e = EnsembleModel(s, null)
        assertArrayEquals(slowAnswer, e.predict(window(), 25f)[0], 0f)
        assertEquals(ModelRole.SLOW, e.lastUsed)
        assertFalse(e.hasFast)
        assertEquals(1, e.fallbackCount)
    }

    @Test
    fun `missing slow model means fast covers every speed, still told the real speed`() {
        val f = fast()
        val e = EnsembleModel(null, f)
        assertArrayEquals(fastAnswer, e.predict(window(), 2f)[0], 0f)
        assertEquals(0.2f, f.lastInput!![0][6], 1e-6f)
        assertEquals(ModelRole.FAST, e.lastUsed)
    }

    @Test
    fun `an ensemble with no models is refused`() {
        assertThrows(IllegalArgumentException::class.java) { EnsembleModel(null, null) }
    }

    @Test
    fun `a throwing fast model falls back to slow for that chunk`() {
        val s = slow()
        val e = EnsembleModel(s, Throwing())
        assertArrayEquals(slowAnswer, e.predict(window(), 20f)[0], 0f)
        assertEquals(ModelRole.SLOW, e.lastUsed)
        assertEquals(1, e.fallbackCount)
    }

    @Test
    fun `a throwing slow model falls back to fast`() {
        val f = fast()
        val e = EnsembleModel(Throwing(), f)
        assertArrayEquals(fastAnswer, e.predict(window(), 1f)[0], 0f)
        assertEquals(ModelRole.FAST, e.lastUsed)
    }

    @Test
    fun `non-finite or wrong-shaped output counts as a failure`() {
        val nan = Returning { n -> Array(n) { floatArrayOf(Float.NaN, 0f) } }
        val short = Returning { n -> Array(n - 1) { floatArrayOf(0f, 0f) } }
        val wide = Returning { n -> Array(n) { floatArrayOf(0f, 0f, 0f) } }
        for (bad in listOf(nan, short, wide)) {
            val s = slow()
            val e = EnsembleModel(s, bad)
            assertArrayEquals(slowAnswer, e.predict(window(), 20f)[0], 0f)
            assertEquals(1, s.calls)
        }
    }

    @Test
    fun `a window the speed channel rejects falls back instead of throwing`() {
        val s = Returning { n -> Array(n) { slowAnswer.copyOf() } }
        val e = EnsembleModel(s, fast())
        val wide = Array(50) { FloatArray(7) }
        assertArrayEquals(slowAnswer, e.predict(wide, 20f)[0], 0f)
        assertEquals(ModelRole.SLOW, e.lastUsed)
        assertEquals(1, e.fallbackCount)
    }

    @Test
    fun `when both models fail the chunk gets zero corrections, not a crash`() {
        val e = EnsembleModel(Throwing(), Throwing())
        val out = e.predict(window(), 10f)
        assertEquals(50, out.size)
        assertTrue(out.all { it.size == 2 && it[0] == 0f && it[1] == 0f })
        assertNull(e.lastUsed)
        assertEquals(1, e.fallbackCount)
    }

    @Test
    fun `a failure in one chunk does not stick to the next`() {
        var failNext = true
        val flaky = object : Predictor {
            override fun predict(window: Array<FloatArray>): Array<FloatArray> {
                if (failNext) { failNext = false; throw IllegalStateException("once") }
                return Array(window.size) { fastAnswer.copyOf() }
            }
        }
        val e = EnsembleModel(slow(), flaky)
        e.predict(window(), 20f)
        assertEquals(ModelRole.SLOW, e.lastUsed)
        e.predict(window(), 20f)
        assertEquals(ModelRole.FAST, e.lastUsed)
    }

    @Test
    fun `close closes both models`() {
        val s = slow(); val f = fast()
        EnsembleModel(s, f).close()
        assertTrue(s.closed); assertTrue(f.closed)
    }

    // ---------------------------------------------------------- loading

    private class FakeSession(val bytes: ByteArray) { var closed = false }

    private val bundled = byteArrayOf(1)
    private val good = byteArrayOf(2)
    private val badShape = byteArrayOf(3)
    private val corrupt = byteArrayOf(4)

    private fun load(
        downloaded: ByteArray?,
        bundledBytes: ByteArray = bundled,
        failures: MutableList<Exception> = mutableListOf(),
        sessions: MutableList<FakeSession> = mutableListOf(),
    ) = ModelFallback.load(
        downloadedVersion = downloaded?.let { 5 },
        downloadedBytes = downloaded?.let { d -> { d } },
        bundledVersion = 1,
        bundledBytes = { bundledBytes },
        open = { b ->
            if (b.contentEquals(corrupt)) throw IllegalArgumentException("not an onnx file")
            FakeSession(b).also { sessions += it }
        },
        check = { s -> require(!s.bytes.contentEquals(badShape)) { "wrong shape" } },
        close = { it.closed = true },
        onDownloadedFailed = { failures += it },
    )

    @Test
    fun `a good download is used`() {
        val failures = mutableListOf<Exception>()
        val r = load(good, failures = failures)
        assertSame(ModelSource.DOWNLOADED, r.source)
        assertEquals(5, r.version)
        assertArrayEquals(good, r.session.bytes)
        assertTrue(failures.isEmpty())
    }

    @Test
    fun `no download means the bundled model`() {
        val r = load(null)
        assertSame(ModelSource.BUNDLED, r.source)
        assertEquals(1, r.version)
    }

    @Test
    fun `a download that will not open falls back to bundled and is reported`() {
        val failures = mutableListOf<Exception>()
        val r = load(corrupt, failures = failures)
        assertSame(ModelSource.BUNDLED, r.source)
        assertEquals(1, failures.size)
    }

    @Test
    fun `a download with the wrong shape is closed, reported, and bundled loads`() {
        val failures = mutableListOf<Exception>()
        val sessions = mutableListOf<FakeSession>()
        val r = load(badShape, failures = failures, sessions = sessions)
        assertSame(ModelSource.BUNDLED, r.source)
        assertEquals(1, failures.size)
        assertTrue(sessions.first { it.bytes.contentEquals(badShape) }.closed)
        assertFalse(r.session.closed)
    }

    @Test
    fun `a download whose bytes cannot be read falls back to bundled`() {
        val failures = mutableListOf<Exception>()
        val r = ModelFallback.load(
            downloadedVersion = 5,
            downloadedBytes = { throw java.io.IOException("gone") },
            bundledVersion = 1,
            bundledBytes = { bundled },
            open = { FakeSession(it) },
            check = {},
            close = {},
            onDownloadedFailed = { failures += it },
        )
        assertSame(ModelSource.BUNDLED, r.source)
        assertEquals(1, failures.size)
    }

    @Test
    fun `a broken bundled model throws and is closed`() {
        val sessions = mutableListOf<FakeSession>()
        assertThrows(IllegalArgumentException::class.java) { load(null, bundledBytes = badShape, sessions = sessions) }
        assertTrue(sessions.single().closed)
        assertThrows(IllegalArgumentException::class.java) { load(null, bundledBytes = corrupt) }
    }
}
