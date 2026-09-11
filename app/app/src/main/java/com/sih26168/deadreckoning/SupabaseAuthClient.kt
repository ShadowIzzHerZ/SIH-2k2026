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
        }
    }

    private fun extractErrorMessage(responseBody: String, httpCode: Int): String =
        runCatching {
            val json = JSONObject(responseBody)
            json.optString("msg", json.optString("message", "HTTP $httpCode"))
        }.getOrDefault("HTTP $httpCode")
}
