package com.sih26168.deadreckoning

/**
 * FusionEngine's only real dependency on BiasCorrectionModel — pulled out
 * so a plain JVM unit test (no device, no ONNX runtime, no Android
 * Context) can exercise FusionEngine's actual integration/ZUPT logic
 * against a fake Predictor instead. See FusionEngineTest.
 */
interface Predictor {
    fun predict(window: Array<FloatArray>): Array<FloatArray>

    /** [startSpeed] is the speed (m/s) the chunk's integration starts
     * from. Only EnsembleModel uses it, to pick a model and fill the fast
     * model's speed channel; everything else ignores it. */
    fun predict(window: Array<FloatArray>, startSpeed: Float): Array<FloatArray> = predict(window)
}
