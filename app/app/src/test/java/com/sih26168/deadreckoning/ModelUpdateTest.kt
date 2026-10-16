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
}
