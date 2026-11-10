package com.sih26168.deadreckoning

import android.content.Context

/** Keeps the last drive's CalibrationPrior across app starts. Live
 * sensors only: replay recordings come from other phones and mounts. */
object CalibrationPriorStore {
    private const val PREFS = "calibration_prior"
    private const val KEY = "prior_json"

    fun load(context: Context): CalibrationPrior? =
        CalibrationPrior.fromJson(context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getString(KEY, null))

    fun save(context: Context, prior: CalibrationPrior) {
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit().putString(KEY, prior.toJson()).apply()
    }

    fun clear(context: Context) {
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit().remove(KEY).apply()
    }
}
