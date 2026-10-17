package com.sih26168.deadreckoning

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.IOException
import java.time.Instant
import java.util.concurrent.TimeUnit

/**
 * Optional account/data-contribution layer talking to the SIH26168 Supabase
 * project (see ../../SUPABASE.md for the schema and why this is a separate
 * database from the AWS telemetry backend). Nothing in here is called by
 * the dead-reckoning pipeline (SensorReader/FusionEngine/BiasCorrectionModel)
 * — this exists purely for: (1) sign-up/sign-in, answering the "what if
 * 1000+ people join" question with real managed Postgres+Auth rather than a
 * bespoke table, and (2) opt-in drive-session recording for future model
 * retraining.
 *
 * Deliberately plain OkHttp + org.json calling Supabase's REST endpoints
 * directly (GoTrue for auth, PostgREST for tables) rather than the
 * supabase-kt SDK — see the Gradle dependency comment for why.
 */
object SupabaseAuthClient {

    // Public by design: Supabase's publishable/anon key is meant to be
    // embedded in client apps. Row Level Security (enabled on every table,
    // see SUPABASE.md) is what actually enforces access control, not
    // keeping this key secret.
    private const val TAG = "SupabaseAuthClient"
    private const val SUPABASE_URL = "https://iekkwudnpvijroqvknwq.supabase.co"
    private const val SUPABASE_ANON_KEY =
        "sb_publishable_g0HHEKwCxphZyVWAFvYqWw_RBATOdJ7"

    private const val PREFS_NAME = "supabase_session_prefs"
    private const val KEY_ACCESS_TOKEN = "access_token"
    private const val KEY_REFRESH_TOKEN = "refresh_token"
    private const val KEY_USER_ID = "user_id"
    // Local mirror of profiles.data_contribution_opt_in — see
    // getCachedConsent's doc for why this exists instead of a network
    // round-trip on every recording stop.
    private const val KEY_CACHED_CONSENT = "cached_data_contribution_opt_in"

    private val client = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(10, TimeUnit.SECONDS)
        .build()
    private val jsonMediaType = "application/json".toMediaType()

    data class Session(val accessToken: String, val refreshToken: String, val userId: String)

    /** Result of a sign-up call: a project with email confirmation enabled
     * (Supabase's default for a new project) returns no session until the
     * user clicks the confirmation link — the caller needs to distinguish
     * that from an immediate session so the UI can say "check your email"
     * instead of silently doing nothing. */
    sealed class SignUpResult {
        data class SignedIn(val session: Session) : SignUpResult()
        object ConfirmationEmailSent : SignUpResult()
    }

    fun getStoredSession(context: Context): Session? {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val accessToken = prefs.getString(KEY_ACCESS_TOKEN, null) ?: return null
        val refreshToken = prefs.getString(KEY_REFRESH_TOKEN, null) ?: return null
        val userId = prefs.getString(KEY_USER_ID, null) ?: return null
        return Session(accessToken, refreshToken, userId)
    }

    fun clearStoredSession(context: Context) {
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE).edit().clear().apply()
    }

    /** Local mirror of profiles.data_contribution_opt_in, written whenever
     * setDataContributionOptIn succeeds — read by MainActivity before every
     * recording upload so that check doesn't need its own network
     * round-trip (and, unlike a network call, fails closed: no cached
     * "true" means no upload, never the reverse). Cleared alongside the
     * rest of the session on sign-out so a fresh sign-in doesn't inherit a
     * stale opt-in from a previous account on this device. */
    fun getCachedConsent(context: Context): Boolean =
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .getBoolean(KEY_CACHED_CONSENT, false)

    private fun setCachedConsent(context: Context, optIn: Boolean) {
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE).edit()
            .putBoolean(KEY_CACHED_CONSENT, optIn)
            .apply()
    }

    private fun storeSession(context: Context, session: Session) {
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE).edit()
            .putString(KEY_ACCESS_TOKEN, session.accessToken)
            .putString(KEY_REFRESH_TOKEN, session.refreshToken)
            .putString(KEY_USER_ID, session.userId)
            .apply()
    }

    /** POST {url}/auth/v1/signup — see SignUpResult's doc for why this
     * doesn't always yield an immediate session. */
    suspend fun signUp(context: Context, email: String, password: String): Result<SignUpResult> =
        withContext(Dispatchers.IO) {
            runCatching {
                val body = JSONObject().apply {
                    put("email", email)
                    put("password", password)
                }.toString().toRequestBody(jsonMediaType)

                val request = Request.Builder()
                    .url("$SUPABASE_URL/auth/v1/signup")
                    .addHeader("apikey", SUPABASE_ANON_KEY)
                    .post(body)
                    .build()

                client.newCall(request).execute().use { response ->
                    val responseBody = response.body?.string().orEmpty()
                    Log.d(TAG, "signUp: HTTP ${response.code}, body=$responseBody")
                    if (!response.isSuccessful) {
                        throw IOException(extractErrorMessage(responseBody, response.code))
                    }
                    val json = JSONObject(responseBody)
                    val accessToken = json.optString("access_token", "")
                    if (accessToken.isEmpty()) {
                        // No access_token in the response == email confirmation is
                        // pending; Supabase still returns a 200 with a `user` object.
                        Log.d(TAG, "signUp: no access_token in response, treating as confirmation-pending")
                        SignUpResult.ConfirmationEmailSent
                    } else {
                        val session = Session(
                            accessToken = accessToken,
                            refreshToken = json.getString("refresh_token"),
                            userId = json.getJSONObject("user").getString("id"),
                        )
                        storeSession(context, session)
                        Log.d(TAG, "signUp: signed in immediately, userId=${session.userId}")
                        SignUpResult.SignedIn(session)
                    }
                }
            }.onFailure { Log.e(TAG, "signUp failed", it) }
        }

    /** POST {url}/auth/v1/token?grant_type=password */
    suspend fun signIn(context: Context, email: String, password: String): Result<Session> =
        withContext(Dispatchers.IO) {
            runCatching {
                val body = JSONObject().apply {
                    put("email", email)
                    put("password", password)
                }.toString().toRequestBody(jsonMediaType)

                val request = Request.Builder()
                    .url("$SUPABASE_URL/auth/v1/token?grant_type=password")
                    .addHeader("apikey", SUPABASE_ANON_KEY)
                    .post(body)
                    .build()

                client.newCall(request).execute().use { response ->
                    val responseBody = response.body?.string().orEmpty()
                    Log.d(TAG, "signIn: HTTP ${response.code}, body=$responseBody")
                    if (!response.isSuccessful) {
                        throw IOException(extractErrorMessage(responseBody, response.code))
                    }
                    val json = JSONObject(responseBody)
                    val session = Session(
                        accessToken = json.getString("access_token"),
                        refreshToken = json.getString("refresh_token"),
                        userId = json.getJSONObject("user").getString("id"),
                    )
                    storeSession(context, session)
                    session
                }
            }.onFailure { Log.e(TAG, "signIn failed", it) }
        }

    /** POST {url}/auth/v1/token?grant_type=refresh_token — trades the
     * session's refresh token for a fresh access token (Supabase's expire
     * after an hour by default). Supabase rotates the refresh token on every
     * use, so the new session is stored straight away; the old refresh token
     * stops working. If GoTrue says the refresh token itself is no longer
     * valid (revoked, already used, signed out elsewhere), the stored
     * session is cleared too: it can never work again, and keeping it made
     * every later upload fail the same way without anything changing.
     * Blocking: call from Dispatchers.IO. */
    fun refreshSession(context: Context, session: Session): Session {
        val body = JSONObject().apply {
            put("refresh_token", session.refreshToken)
        }.toString().toRequestBody(jsonMediaType)

        val request = Request.Builder()
            .url("$SUPABASE_URL/auth/v1/token?grant_type=refresh_token")
            .addHeader("apikey", SUPABASE_ANON_KEY)
            .post(body)
            .build()

        client.newCall(request).execute().use { response ->
            val responseBody = response.body?.string().orEmpty()
            if (!response.isSuccessful) {
                // Only if it's still the stored session, so a newer sign-in
                // or refresh isn't thrown away.
                if (isRevokedRefreshToken(response.code, responseBody) &&
                    getStoredSession(context)?.refreshToken == session.refreshToken
                ) {
                    Log.w(TAG, "refresh token rejected, clearing the stored session")
                    clearStoredSession(context)
                }
                throw SupabaseHttpException(response.code, extractErrorMessage(responseBody, response.code))
            }
            val refreshed = parseSession(responseBody)
            storeSession(context, refreshed)
            return refreshed
        }
    }

    /** Upserts this install's DeviceIdentity.get(context) value into the
     * `devices` table under the signed-in user — RLS's devices_insert_own/
     * devices_update_own policies require owner_id == auth.uid(), which is
     * exactly the session's own userId, so a spoofed owner_id would just be
     * rejected server-side regardless of what's sent here. */
    suspend fun registerDevice(
        session: Session,
        deviceId: String,
        vehicleLabel: String?,
    ): Result<Unit> = withContext(Dispatchers.IO) {
        runCatching {
            val body = JSONObject().apply {
                put("owner_id", session.userId)
                put("device_id", deviceId)
                if (vehicleLabel != null) put("vehicle_label", vehicleLabel)
            }.toString().toRequestBody(jsonMediaType)

            val request = Request.Builder()
                .url("$SUPABASE_URL/rest/v1/devices?on_conflict=device_id")
                .addHeader("apikey", SUPABASE_ANON_KEY)
                .addHeader("Authorization", "Bearer ${session.accessToken}")
                .addHeader("Prefer", "resolution=merge-duplicates,return=minimal")
                .post(body)
                .build()

            client.newCall(request).execute().use { response ->
                if (!response.isSuccessful) {
                    val responseBody = response.body?.string().orEmpty()
                    throw IOException(extractErrorMessage(responseBody, response.code))
                }
            }
        }
    }

    /** PATCH {url}/rest/v1/profiles?id=eq.{userId} — flips the explicit,
     * revocable consent flag gating whether future drive_sessions may be
     * used for retraining. Never called automatically; only ever in
     * response to a real user action in AuthActivity's UI. */
    suspend fun setDataContributionOptIn(
        context: Context,
        session: Session,
        optIn: Boolean,
    ): Result<Unit> = withContext(Dispatchers.IO) {
        runCatching {
            val body = JSONObject().apply {
                put("data_contribution_opt_in", optIn)
            }.toString().toRequestBody(jsonMediaType)

            val request = Request.Builder()
                .url("$SUPABASE_URL/rest/v1/profiles?id=eq.${session.userId}")
                .addHeader("apikey", SUPABASE_ANON_KEY)
                .addHeader("Authorization", "Bearer ${session.accessToken}")
                .addHeader("Prefer", "return=minimal")
                .patch(body)
                .build()

            client.newCall(request).execute().use { response ->
                if (!response.isSuccessful) {
                    val responseBody = response.body?.string().orEmpty()
                    throw IOException(extractErrorMessage(responseBody, response.code))
                }
            }
            // Only mirrored locally once the server call actually
            // succeeded — see getCachedConsent's doc for why this can't
            // just optimistically cache whatever the UI checkbox said.
            setCachedConsent(context, optIn)
        }
    }

    /** POST {url}/rest/v1/drive_sessions — one row per recorded trip, see
     * SUPABASE.md's schema section. `contributeToTraining` is a per-session
     * snapshot of the user's consent at upload time (matching the
     * `contribute_to_training` column's own doc comment), not a live join
     * back to `profiles`; the caller (MainActivity) already gated calling
     * this at all on getCachedConsent(), so it's passed through as-is
     * rather than re-checked here. `raw_data_ref`/`used_for_training` are
     * left at their column defaults (null/false) — nothing yet uploads the
     * actual raw IMU log this row would point at (see SUPABASE.md's
     * "what's NOT built yet" #4). A 401 (the stored access token expired)
     * refreshes the session once and retries, see refreshSession. */
    suspend fun recordDriveSession(
        context: Context,
        session: Session,
        deviceId: String,
        startedAt: Instant,
        endedAt: Instant,
        startLat: Double?,
        startLon: Double?,
        endLat: Double?,
        endLon: Double?,
        hadBlackout: Boolean,
        contributeToTraining: Boolean,
    ): Result<Unit> = withContext(Dispatchers.IO) {
        runCatching {
            val body = JSONObject().apply {
                put("device_id", deviceId)
                put("started_at", startedAt.toString())
                put("ended_at", endedAt.toString())
                put("start_lat", startLat)
                put("start_lon", startLon)
                put("end_lat", endLat)
                put("end_lon", endLon)
                put("had_blackout", hadBlackout)
                put("contribute_to_training", contributeToTraining)
            }.toString().toRequestBody(jsonMediaType)

            retryOnceOnUnauthorized(session, refresh = { refreshSession(context, it) }) { current ->
                val request = Request.Builder()
                    .url("$SUPABASE_URL/rest/v1/drive_sessions")
                    .addHeader("apikey", SUPABASE_ANON_KEY)
                    .addHeader("Authorization", "Bearer ${current.accessToken}")
                    .addHeader("Prefer", "return=minimal")
                    .post(body)
                    .build()

                client.newCall(request).execute().use { response ->
                    if (!response.isSuccessful) {
                        val responseBody = response.body?.string().orEmpty()
                        throw SupabaseHttpException(response.code, extractErrorMessage(responseBody, response.code))
                    }
                }
            }
        }.onFailure { Log.e(TAG, "recordDriveSession failed", it) }
    }

    private fun extractErrorMessage(responseBody: String, httpCode: Int): String =
        runCatching {
            val json = JSONObject(responseBody)
            json.optString("msg", json.optString("message", "HTTP $httpCode"))
        }.getOrDefault("HTTP $httpCode")
}

/** A non-2xx Supabase response, keeping the HTTP status so an expired
 * access token (401) can be told apart from other failures. */
internal class SupabaseHttpException(val code: Int, message: String) : IOException(message)

/** True when a refresh_token grant failed because the refresh token itself
 * is dead: HTTP 400/401 with `"error": "invalid_grant"` (older GoTrue) or an
 * `error_code` saying the refresh token, session or user is gone (newer
 * GoTrue). Network errors, 5xx and anything else keep the session so a
 * later refresh can still succeed. */
internal fun isRevokedRefreshToken(httpCode: Int, responseBody: String): Boolean {
    if (httpCode != 400 && httpCode != 401) return false
    val json = runCatching { JSONObject(responseBody) }.getOrNull() ?: return false
    val errorCode = json.optString("error_code")
    return json.optString("error") == "invalid_grant" ||
        errorCode.startsWith("refresh_token_") ||
        errorCode in setOf("session_not_found", "session_expired", "user_not_found", "user_banned")
}

/** A GoTrue token response (sign-in or refresh) as a Session. */
internal fun parseSession(responseBody: String): SupabaseAuthClient.Session {
    val json = JSONObject(responseBody)
    return SupabaseAuthClient.Session(
        accessToken = json.getString("access_token"),
        refreshToken = json.getString("refresh_token"),
        userId = json.getJSONObject("user").getString("id"),
    )
}

/** Runs [call] with [session]; if it fails with a 401, gets a new session
 * from [refresh] and runs [call] once more with that. Any other failure,
 * or a second 401, is thrown as is. */
internal fun <T> retryOnceOnUnauthorized(
    session: SupabaseAuthClient.Session,
    refresh: (SupabaseAuthClient.Session) -> SupabaseAuthClient.Session,
    call: (SupabaseAuthClient.Session) -> T,
): T =
    try {
        call(session)
    } catch (e: SupabaseHttpException) {
        if (e.code != 401) throw e
        call(refresh(session))
    }
