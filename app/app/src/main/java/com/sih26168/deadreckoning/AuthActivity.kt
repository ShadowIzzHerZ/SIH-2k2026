package com.sih26168.deadreckoning

import android.content.Intent
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch

/**
 * Optional sign-up/sign-in screen for the Supabase-backed account layer
 * (SupabaseAuthClient.kt, see ../../../../../../SUPABASE.md). Entirely
 * separate from the dead-reckoning demo: "Continue without an account"
 * always works and never touches the network, since the pipeline itself
 * (SensorReader -> FusionEngine -> BiasCorrectionModel) has zero account
 * dependency by design.
 *
 * On a successful sign-in or sign-up, registers this install's stable
 * DeviceIdentity under the account and applies the consent checkbox's
 * state, then returns to MainActivity.
 */
class AuthActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "AuthActivity"

        fun launch(activity: AppCompatActivity) {
            activity.startActivity(Intent(activity, AuthActivity::class.java))
        }
    }

    private lateinit var emailField: EditText
    private lateinit var passwordField: EditText
    private lateinit var consentCheckbox: CheckBox
    private lateinit var statusText: TextView
    private lateinit var authProgress: ProgressBar
    private lateinit var signInButton: Button
    private lateinit var signUpButton: Button

    // No lifecycleScope here — that needs androidx.lifecycle:lifecycle-
    // runtime-ktx, an extra dependency this module doesn't otherwise need
    // (see the OkHttp-vs-supabase-kt dependency comment in build.gradle.kts
    // for the same reasoning). kotlinx-coroutines-android is already a
    // dependency, so a plain Activity-scoped CoroutineScope does the job.
    private val activityScope = CoroutineScope(Dispatchers.Main + Job())

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_auth)

        emailField = findViewById(R.id.emailField)
        passwordField = findViewById(R.id.passwordField)
        consentCheckbox = findViewById(R.id.consentCheckbox)
        statusText = findViewById(R.id.statusText)
        authProgress = findViewById(R.id.authProgress)
        signInButton = findViewById(R.id.signInButton)
        signUpButton = findViewById(R.id.signUpButton)
        val skipButton: TextView = findViewById(R.id.skipButton)

        // Already signed in from a previous launch — nothing to do here.
        if (SupabaseAuthClient.getStoredSession(this) != null) {
            finish()
            return
        }

        signInButton.setOnClickListener { Log.d(TAG, "signInButton clicked"); submit(isSignUp = false) }
        signUpButton.setOnClickListener { Log.d(TAG, "signUpButton clicked"); submit(isSignUp = true) }
        skipButton.setOnClickListener { Log.d(TAG, "skipButton clicked, finishing without auth"); finish() }
    }

    private fun submit(isSignUp: Boolean) {
        val email = emailField.text.toString().trim()
        val password = passwordField.text.toString()
        if (email.isEmpty() || password.isEmpty()) {
            showStatus(getString(R.string.auth_error_empty_fields))
            return
        }

        setLoading(true)
        activityScope.launch {
            val outcome = if (isSignUp) {
                handleSignUp(email, password)
            } else {
                handleSignIn(email, password)
            }
            setLoading(false)
            outcome.fold(
                onSuccess = { finished ->
                    Log.d(TAG, "submit(isSignUp=$isSignUp) succeeded, finished=$finished")
                    if (finished) finish()
                },
                onFailure = { error ->
                    Log.e(TAG, "submit(isSignUp=$isSignUp) failed", error)
                    showStatus(error.message ?: "Something went wrong")
                },
            )
        }
    }

    /** Returns Result<finished> — true if the flow is complete and the
     * activity should close, false if it succeeded but needs the user to
     * take another step first (e.g. confirm their email). */
    private suspend fun handleSignUp(email: String, password: String): Result<Boolean> =
        SupabaseAuthClient.signUp(this, email, password).mapCatching { result ->
            when (result) {
                is SupabaseAuthClient.SignUpResult.ConfirmationEmailSent -> {
                    showStatus(getString(R.string.auth_confirmation_sent))
                    false
                }
                is SupabaseAuthClient.SignUpResult.SignedIn -> {
                    onAuthenticated(result.session)
                    true
                }
            }
        }

    private suspend fun handleSignIn(email: String, password: String): Result<Boolean> =
        SupabaseAuthClient.signIn(this, email, password).mapCatching { session ->
            onAuthenticated(session)
            true
        }

    private suspend fun onAuthenticated(session: SupabaseAuthClient.Session) {
        val deviceId = DeviceIdentity.get(this)
        // Best-effort: device registration/consent failing shouldn't strand
        // the user on this screen after a real, successful sign-in/sign-up.
        SupabaseAuthClient.registerDevice(session, deviceId, vehicleLabel = null)
        SupabaseAuthClient.setDataContributionOptIn(session, consentCheckbox.isChecked)
        showStatus(getString(R.string.auth_signed_in))
    }

    private fun setLoading(loading: Boolean) {
        authProgress.visibility = if (loading) View.VISIBLE else View.GONE
        signInButton.isEnabled = !loading
        signUpButton.isEnabled = !loading
    }

    private fun showStatus(message: String) {
        statusText.text = message
        statusText.visibility = View.VISIBLE
        Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
    }

    override fun onDestroy() {
        super.onDestroy()
        activityScope.cancel()
    }
}
