package com.sih26168.deadreckoning

import android.content.Intent
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity

/**
 * One-time disclosure screen, shown before MainActivity ever requests the
 * OS location permission — see PrivacyConsent.kt and strings.xml's
 * privacy_* comment for why this exists as its own screen and why each
 * point on it is checkable against real app behavior. MainActivity.onCreate()
 * checks PrivacyConsent.hasConsented() before doing anything else and
 * redirects here if it's false; this screen's only path forward sets that
 * flag and launches MainActivity fresh, mirroring the same
 * check-and-redirect pattern AuthActivity already uses for a stored
 * session.
 */
class PrivacyConsentActivity : AppCompatActivity() {

    companion object {
        fun launch(activity: AppCompatActivity) {
            activity.startActivity(Intent(activity, PrivacyConsentActivity::class.java))
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_privacy_consent)

        findViewById<android.widget.Button>(R.id.privacyAgreeButton).setOnClickListener {
            PrivacyConsent.setConsented(this)
            startActivity(Intent(this, MainActivity::class.java))
            finish()
        }
    }
}
