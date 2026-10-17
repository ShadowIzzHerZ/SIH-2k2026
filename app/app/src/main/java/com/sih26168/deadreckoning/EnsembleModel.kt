package com.sih26168.deadreckoning

/**
 * Which network handles a chunk, and the 7th input channel the fast one
 * needs. Matches experiments/drift/run_duo.py (drift-experiments branch)
 * and src/export_ensemble.py exactly; change both only with a retrain.
 */
object SpeedGate {
    /** At or above this start speed the FAST model runs, below it SLOW. */
    const val GATE_MPS = 6f

    /** The FAST model's 7th channel is the start speed divided by this. */
    const val SPEED_SCALE = 10f

    /** A non-finite speed means the estimate is broken; SLOW is the safer
     * guess because it doesn't read the speed at all. */
    fun useFast(startSpeed: Float): Boolean = startSpeed.isFinite() && startSpeed >= GATE_MPS

    /** [window] rows of 6 with startSpeed / SPEED_SCALE appended to every
     * row, the same value on every sample (as in training). */
    fun withSpeedChannel(window: Array<FloatArray>, startSpeed: Float): Array<FloatArray> {
        val s = if (startSpeed.isFinite()) startSpeed / SPEED_SCALE else 0f
        return Array(window.size) { i ->
            val row = window[i]
            require(row.size == 6) { "expected 6 channels, got ${row.size}" }
            FloatArray(7).also {
                System.arraycopy(row, 0, it, 0, 6)
                it[6] = s
            }
        }
    }
}

/** Where a role's model came from at load time. */
enum class ModelSource { DOWNLOADED, BUNDLED }

/**
 * The load order for one role: a verified download if there is one and it
 * opens with the right shapes, else the copy bundled in the APK. Generic
 * over the session type so ModelFallbackTest can drive it with fakes; the
 * real caller is BiasCorrectionModel with ONNX Runtime sessions.
 */
object ModelFallback {
    class Loaded<S>(val session: S, val version: Int, val source: ModelSource)

    /**
     * [open] turns model bytes into a session, [check] throws if the
     * session's shapes are wrong, [close] releases a session that failed
     * its check. [onDownloadedFailed] is told when a download had to be
     * dropped, so the caller can delete it and not fetch it again. Throws
     * only if the bundled model itself can't be opened or checked.
     */
    fun <S> load(
        downloadedVersion: Int?,
        downloadedBytes: (() -> ByteArray)?,
        bundledVersion: Int,
        bundledBytes: () -> ByteArray,
        open: (ByteArray) -> S,
        check: (S) -> Unit,
        close: (S) -> Unit,
        onDownloadedFailed: (Exception) -> Unit,
    ): Loaded<S> {
        if (downloadedVersion != null && downloadedBytes != null) {
            var session: S? = null
            try {
                session = open(downloadedBytes())
                check(session)
                return Loaded(session, downloadedVersion, ModelSource.DOWNLOADED)
            } catch (e: Exception) {
                session?.let { runCatching { close(it) } }
                onDownloadedFailed(e)
            }
        }
        val session = open(bundledBytes())
        try {
            check(session)
        } catch (e: Exception) {
            runCatching { close(session) }
            throw e
        }
        return Loaded(session, bundledVersion, ModelSource.BUNDLED)
    }
}

/**
 * The two-model, speed-gated ensemble: one 5 s chunk goes to [fast] when
 * the speed it starts from is at or above SpeedGate.GATE_MPS, to [slow]
 * otherwise. [fast] takes the 7-channel window (SpeedGate.withSpeedChannel
 * is applied here, so callers always pass the plain 6 channels).
 *
 * Either model may be missing (it failed to load even from the APK); the
 * other then handles every speed. If the chosen model throws or returns
 * something unusable for a chunk, the other one is tried, and if neither
 * works the chunk gets zero corrections, which is plain strapdown
 * integration: worse, but never a crash mid-drive.
 */
class EnsembleModel(
    private val slow: Predictor?,
    private val fast: Predictor?,
) : Predictor, AutoCloseable {

    init {
        require(slow != null || fast != null) { "the ensemble needs at least one model" }
    }

    /** Which role produced the last chunk's corrections, null if neither
     * did (zeros). For logs and tests. */
    @Volatile var lastUsed: ModelRole? = null
        private set

    /** Chunks where the gated model failed and another answer was used. */
    @Volatile var fallbackCount: Int = 0
        private set

    /** Chunks answered by each role, and by neither (zeros). For the
     * on-device self-check (DriveSelfCheck). */
    @Volatile var slowChunks: Int = 0
        private set
    @Volatile var fastChunks: Int = 0
        private set
    @Volatile var zeroChunks: Int = 0
        private set

    /** The start speed the last chunk was gated on. */
    @Volatile var lastStartSpeed: Float = 0f
        private set

    val chunks: Int get() = slowChunks + fastChunks + zeroChunks

    val hasSlow: Boolean get() = slow != null
    val hasFast: Boolean get() = fast != null

    /** No speed known: the slow model, which doesn't need one. */
    override fun predict(window: Array<FloatArray>): Array<FloatArray> = predict(window, 0f)

    override fun predict(window: Array<FloatArray>, startSpeed: Float): Array<FloatArray> {
        lastStartSpeed = startSpeed
        val preferred = if (SpeedGate.useFast(startSpeed)) ModelRole.FAST else ModelRole.SLOW
        val order = if (preferred == ModelRole.FAST) listOf(ModelRole.FAST, ModelRole.SLOW)
        else listOf(ModelRole.SLOW, ModelRole.FAST)
        for (role in order) {
            val out = run(role, window, startSpeed) ?: continue
            if (role != preferred) fallbackCount++
            lastUsed = role
            if (role == ModelRole.FAST) fastChunks++ else slowChunks++
            return out
        }
        fallbackCount++
        lastUsed = null
        zeroChunks++
        return Array(window.size) { FloatArray(2) }
    }

    private fun run(role: ModelRole, window: Array<FloatArray>, startSpeed: Float): Array<FloatArray>? {
        val model = (if (role == ModelRole.FAST) fast else slow) ?: return null
        // withSpeedChannel is inside the try too: a window it rejects must
        // fall back like any other failure, not crash the tick.
        val out = try {
            val input = if (role == ModelRole.FAST) SpeedGate.withSpeedChannel(window, startSpeed) else window
            model.predict(input)
        } catch (e: Exception) {
            return null
        }
        val usable = out.size == window.size && out.all { it.size == 2 && it.all(Float::isFinite) }
        return if (usable) out else null
    }

    override fun close() {
        (slow as? AutoCloseable)?.let { runCatching { it.close() } }
        (fast as? AutoCloseable)?.let { runCatching { it.close() } }
    }
}
