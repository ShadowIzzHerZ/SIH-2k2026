package com.sih26168.deadreckoning

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The accept/reject rules ModelUpdater applies to a published model
 * before it can replace the bundled one. The ONNX Runtime side (opening
 * the file, the test inference) needs a device and was checked on the
 * emulator instead.
 */
class ModelUpdateTest {

    private val sha = "746248cb70d4816167db67cae940612b999c5ad16542fb8240fd9bb79713c82e"

    private fun manifest(
        version: Any = 3,
        file: String = "dead_reckoning_model_v3.onnx",
        sha256: String = sha,
        size: Any = 860807,
    ) = """{"version": $version, "file": "$file", "sha256": "$sha256", "size_bytes": $size}"""

    @Test
    fun parsesAValidManifest() {
        val m = ModelManifest.parse(manifest())
        assertEquals(3, m.version)
        assertEquals("dead_reckoning_model_v3.onnx", m.file)
        assertEquals(sha, m.sha256)
        assertEquals(860807L, m.sizeBytes)
    }

    @Test
    fun uppercaseChecksumIsNormalised() {
        assertEquals(sha, ModelManifest.parse(manifest(sha256 = sha.uppercase())).sha256)
    }

    @Test
    fun rejectsFileNamesThatAreNotAPlainOnnxName() {
        for (bad in listOf("../secrets.onnx", "sub/dir.onnx", "https://evil.example/m.onnx", "model.bin", "")) {
            assertThrows(bad, IllegalArgumentException::class.java) { ModelManifest.parse(manifest(file = bad)) }
        }
    }

    @Test
    fun rejectsBadChecksumsVersionsAndSizes() {
        assertThrows(IllegalArgumentException::class.java) { ModelManifest.parse(manifest(sha256 = "abc")) }
        assertThrows(IllegalArgumentException::class.java) { ModelManifest.parse(manifest(sha256 = "z".repeat(64))) }
        assertThrows(IllegalArgumentException::class.java) { ModelManifest.parse(manifest(version = 0)) }
        assertThrows(IllegalArgumentException::class.java) { ModelManifest.parse(manifest(size = 0)) }
        assertThrows(IllegalArgumentException::class.java) {
            ModelManifest.parse(manifest(size = ModelManifest.MAX_MODEL_BYTES + 1))
        }
    }

    @Test
    fun rejectsMalformedJson() {
        assertThrows(Exception::class.java) { ModelManifest.parse("not json") }
        assertThrows(Exception::class.java) { ModelManifest.parse("""{"version": 3}""") }
    }

    @Test
    fun acceptsOnlyTheExactShapesFusionEngineUses() {
        val ok = longArrayOf(1, 50, 6)
        val okOut = longArrayOf(1, 50, 2)
        assertTrue(ModelChecks.shapesCompatible(listOf(ok), listOf(okOut)))
        // 7 channels, a different window length, a dynamic dim, an extra
        // output, a different output width: all rejected.
        assertFalse(ModelChecks.shapesCompatible(listOf(longArrayOf(1, 50, 7)), listOf(okOut)))
        assertFalse(ModelChecks.shapesCompatible(listOf(longArrayOf(1, 60, 6)), listOf(longArrayOf(1, 60, 2))))
        assertFalse(ModelChecks.shapesCompatible(listOf(longArrayOf(1, -1, 6)), listOf(okOut)))
        assertFalse(ModelChecks.shapesCompatible(listOf(ok), listOf(okOut, okOut)))
        assertFalse(ModelChecks.shapesCompatible(listOf(ok), listOf(longArrayOf(1, 50, 3))))
        assertFalse(ModelChecks.shapesCompatible(emptyList(), listOf(okOut)))
    }

    @Test
    fun downloadsOnlyNewerVersionsThatHaveNotFailedBefore() {
        assertTrue(ModelChecks.shouldDownload(3, 2, null))
        assertFalse(ModelChecks.shouldDownload(2, 2, null))
        assertFalse(ModelChecks.shouldDownload(1, 2, null))
        assertFalse(ModelChecks.shouldDownload(3, 2, 3))
        assertTrue(ModelChecks.shouldDownload(4, 2, 3))
    }

    @Test
    fun sha256MatchesAKnownValue() {
        assertEquals(
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            ModelChecks.sha256Hex(ByteArray(0)),
        )
    }

    private fun entry(version: Int, file: String, size: Int = 860807) =
        """{"version": $version, "file": "$file", "sha256": "$sha", "size_bytes": $size}"""

    @Test
    fun ensembleManifestReadsEachRoleWithItsOwnVersionAndChecksum() {
        val json = """{"version": 3, "file": "dead_reckoning_model_v3.onnx", "sha256": "$sha", "size_bytes": 860807,
            "models": {"slow": ${entry(2, "model_slow_v2.onnx")}, "fast": ${entry(4, "model_fast_v4.onnx", 861447)}}}"""
        val m = EnsembleManifest.parse(json)
        assertEquals(setOf(ModelRole.SLOW, ModelRole.FAST), m.keys)
        assertEquals(2, m[ModelRole.SLOW]!!.version)
        assertEquals("model_fast_v4.onnx", m[ModelRole.FAST]!!.file)
        assertEquals(861447L, m[ModelRole.FAST]!!.sizeBytes)
        // The legacy top-level fields older apps read still parse too.
        assertEquals(3, ModelManifest.parse(json).version)
    }

    @Test
    fun ensembleManifestWithoutModelsOrWithOneRoleOnly() {
        assertTrue(EnsembleManifest.parse(manifest()).isEmpty())
        val one = EnsembleManifest.parse("""{"models": {"fast": ${entry(2, "model_fast_v2.onnx")}}}""")
        assertEquals(setOf(ModelRole.FAST), one.keys)
    }

    @Test
    fun ensembleManifestIgnoresUnknownRolesButRejectsABadKnownOne() {
        val unknown = EnsembleManifest.parse(
            """{"models": {"turbo": {"anything": 1}, "slow": ${entry(2, "model_slow_v2.onnx")}}}""",
        )
        assertEquals(setOf(ModelRole.SLOW), unknown.keys)
        assertThrows(IllegalArgumentException::class.java) {
            EnsembleManifest.parse("""{"models": {"slow": ${entry(2, "../model_slow.onnx")}}}""")
        }
        assertThrows(Exception::class.java) { EnsembleManifest.parse("""{"models": {"fast": {"version": 2}}}""") }
        assertThrows(Exception::class.java) { EnsembleManifest.parse("""{"models": {"fast": "model_fast.onnx"}}""") }
    }

    @Test
    fun eachRoleAcceptsOnlyItsOwnInputWidth() {
        val out = listOf(longArrayOf(1, 50, 2))
        val six = listOf(longArrayOf(1, 50, 6))
        val seven = listOf(longArrayOf(1, 50, 7))
        assertEquals(6, ModelRole.SLOW.channels)
        assertEquals(7, ModelRole.FAST.channels)
        assertTrue(ModelChecks.shapesCompatible(six, out, ModelRole.SLOW))
        assertFalse(ModelChecks.shapesCompatible(seven, out, ModelRole.SLOW))
        assertTrue(ModelChecks.shapesCompatible(seven, out, ModelRole.FAST))
        assertFalse(ModelChecks.shapesCompatible(six, out, ModelRole.FAST))
        assertFalse(ModelChecks.shapesCompatible(listOf(longArrayOf(1, 60, 7)), out, ModelRole.FAST))
        assertFalse(ModelChecks.shapesCompatible(seven, listOf(longArrayOf(1, 50, 3)), ModelRole.FAST))
    }

    @Test
    fun rolesHaveDistinctKeysAndAssets() {
        assertEquals(ModelRole.values().size, ModelRole.values().map { it.key }.toSet().size)
        assertEquals(ModelRole.values().size, ModelRole.values().map { it.asset }.toSet().size)
        assertEquals(ModelRole.SLOW, ModelRole.fromKey("slow"))
        assertEquals(null, ModelRole.fromKey("SLOW"))
    }

    @Test
    fun outputCheckNeedsFiftyFiniteRowsOfTwo() {
        assertTrue(ModelChecks.outputUsable(Array(50) { floatArrayOf(0f, 1f) }))
        assertFalse(ModelChecks.outputUsable(Array(49) { floatArrayOf(0f, 1f) }))
        assertFalse(ModelChecks.outputUsable(Array(50) { floatArrayOf(0f) }))
        assertFalse(ModelChecks.outputUsable(Array(50) { floatArrayOf(0f, Float.NaN) }))
    }
}
