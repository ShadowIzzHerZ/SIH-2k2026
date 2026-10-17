package com.sih26168.deadreckoning

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob

/**
 * Process-lifetime scope for short background work that has to finish
 * even if the Activity that started it doesn't, such as a drive-session
 * upload started as a recording stops. An Activity's own scope is
 * cancelled in onDestroy, which also runs when the Activity is recreated
 * or finished, so work launched there is lost mid-request.
 *
 * SupervisorJob: one failed child doesn't cancel the others. Work
 * launched here must not hold a reference to an Activity or View; pass
 * values or applicationContext instead.
 */
object AppScope : CoroutineScope by CoroutineScope(SupervisorJob() + Dispatchers.Default)
