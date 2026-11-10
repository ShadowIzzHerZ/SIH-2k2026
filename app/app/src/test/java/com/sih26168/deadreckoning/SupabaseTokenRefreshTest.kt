package com.sih26168.deadreckoning

import java.io.IOException
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

/** The refresh-and-retry-once rule SupabaseAuthClient applies to uploads. */
class SupabaseTokenRefreshTest {

    private val expired = SupabaseAuthClient.Session("old-access", "old-refresh", "user-1")
    private val fresh = SupabaseAuthClient.Session("new-access", "new-refresh", "user-1")

    @Test
    fun `a successful call does not refresh`() {
        var refreshes = 0
        val result = retryOnceOnUnauthorized(expired, refresh = { refreshes++; fresh }) { it.accessToken }
        assertEquals("old-access", result)
        assertEquals(0, refreshes)
    }

    @Test
    fun `a 401 refreshes once and retries with the new token`() {
        val tokensUsed = mutableListOf<String>()
        var refreshes = 0
        val result = retryOnceOnUnauthorized(expired, refresh = { refreshes++; fresh }) { s ->
            tokensUsed.add(s.accessToken)
            if (s.accessToken == "old-access") throw SupabaseHttpException(401, "JWT expired")
            "ok"
        }
        assertEquals("ok", result)
        assertEquals(1, refreshes)
        assertEquals(listOf("old-access", "new-access"), tokensUsed)
    }

    @Test
    fun `a second 401 is not retried again`() {
        var calls = 0
        val e = assertThrows(SupabaseHttpException::class.java) {
            retryOnceOnUnauthorized(expired, refresh = { fresh }) {
                calls++
                throw SupabaseHttpException(401, "JWT expired")
            }
        }
        assertEquals(401, e.code)
        assertEquals(2, calls)
    }

    @Test
    fun `other failures are not retried`() {
        var refreshes = 0
        var calls = 0
        assertThrows(SupabaseHttpException::class.java) {
            retryOnceOnUnauthorized(expired, refresh = { refreshes++; fresh }) {
                calls++
                throw SupabaseHttpException(403, "row-level security")
            }
        }
        assertThrows(IOException::class.java) {
            retryOnceOnUnauthorized(expired, refresh = { refreshes++; fresh }) { throw IOException("offline") }
        }
        assertEquals(0, refreshes)
        assertEquals(1, calls)
    }

    @Test
    fun `a failed refresh surfaces the refresh error`() {
        val e = assertThrows(SupabaseHttpException::class.java) {
            retryOnceOnUnauthorized(
                expired,
                refresh = { throw SupabaseHttpException(400, "Invalid Refresh Token") },
            ) { throw SupabaseHttpException(401, "JWT expired") }
        }
        assertEquals(400, e.code)
    }

    @Test
    fun `parses a GoTrue token response`() {
        val body = """{"access_token":"a1","token_type":"bearer","expires_in":3600,
            "refresh_token":"r1","user":{"id":"u1","email":"x@example.com"}}"""
        assertEquals(SupabaseAuthClient.Session("a1", "r1", "u1"), parseSession(body))
    }

    @Test
    fun `a revoked or reused refresh token is recognised`() {
        // Older GoTrue: OAuth-style invalid_grant.
        assertTrue(isRevokedRefreshToken(400,
            """{"error":"invalid_grant","error_description":"Invalid Refresh Token: Already Used"}"""))
        // Newer GoTrue: error_code.
        assertTrue(isRevokedRefreshToken(400,
            """{"code":400,"error_code":"refresh_token_not_found","msg":"Invalid Refresh Token: Refresh Token Not Found"}"""))
        assertTrue(isRevokedRefreshToken(401, """{"code":401,"error_code":"session_not_found","msg":"Session not found"}"""))
    }

    @Test
    fun `other refresh failures keep the session`() {
        // Server errors, rate limits, other 400s and unparsable bodies.
        assertFalse(isRevokedRefreshToken(500, """{"error":"invalid_grant"}"""))
        assertFalse(isRevokedRefreshToken(503, ""))
        assertFalse(isRevokedRefreshToken(400, "<html>Bad gateway</html>"))
        assertFalse(isRevokedRefreshToken(429, """{"error_code":"over_request_rate_limit"}"""))
        assertFalse(isRevokedRefreshToken(400, """{"error_code":"validation_failed","msg":"bad json"}"""))
    }
}
