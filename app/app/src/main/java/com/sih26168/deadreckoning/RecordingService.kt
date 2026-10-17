package com.sih26168.deadreckoning

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.os.PowerManager

/**
 * Keeps a dev-mode recording alive while the app is in the background or
 * the screen is off. It does no work of its own: the recording pipeline
 * stays exactly where it is (MainActivity's 10Hz tick loop, SensorReader,
 * LocationReader, DevRecorder). Without this, Android stops delivering
 * sensor events and throttles location updates for a backgrounded app,
 * and a sleeping CPU stops the main-thread tick Handler, so a recording
 * silently froze the moment the phone was pocketed. Being a foreground
 * service (type "location", since the recording logs GPS) plus holding a
 * partial wake lock is what keeps all of that running.
 *
 * Started/stopped by MainActivity exactly where DevRecorder.start()/stop()
 * already are — see start() and stop() below.
 */
class RecordingService : Service() {

    private var wakeLock: PowerManager.WakeLock? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val notification = buildNotification()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(NOTIFICATION_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_LOCATION)
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }
        if (wakeLock == null) {
            val pm = getSystemService(Context.POWER_SERVICE) as PowerManager
            wakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "DeadReckoning:recording").apply {
                setReferenceCounted(false)
                acquire()
            }
        }
        // Not STICKY: if the process dies, the Activity's pipeline that
        // this service exists to keep alive is gone too, so there's
        // nothing to restart into.
        return START_NOT_STICKY
    }

    override fun onDestroy() {
        wakeLock?.release()
        wakeLock = null
        super.onDestroy()
    }

    private fun buildNotification(): Notification {
        val nm = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        nm.createNotificationChannel(
            NotificationChannel(CHANNEL_ID, getString(R.string.recording_channel_name), NotificationManager.IMPORTANCE_LOW)
        )
        // The launcher intent brings the existing task (and its running
        // MainActivity) to the front rather than starting a second one.
        val open = packageManager.getLaunchIntentForPackage(packageName)?.let {
            PendingIntent.getActivity(this, 0, it, PendingIntent.FLAG_IMMUTABLE)
        }
        return Notification.Builder(this, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_notification_recording)
            .setContentTitle(getString(R.string.recording_notification_title))
            .setContentText(getString(R.string.recording_notification_text))
            .setContentIntent(open)
            .setOngoing(true)
            .build()
    }

    companion object {
        private const val CHANNEL_ID = "recording"
        private const val NOTIFICATION_ID = 1

        /** Call from the foreground Activity only (Android 12+ refuses to
         * start a foreground service from the background). Returns false,
         * starting nothing, when RecordingForeground.canStart says Android
         * would reject the start — the recording then still runs, just
         * only while the app is on screen, as it always did. */
        fun start(context: Context, hasFineLocation: Boolean, hasCoarseLocation: Boolean): Boolean {
            if (!RecordingForeground.canStart(hasFineLocation, hasCoarseLocation)) return false
            context.startForegroundService(Intent(context, RecordingService::class.java))
            return true
        }

        fun stop(context: Context) {
            context.stopService(Intent(context, RecordingService::class.java))
        }
    }
}
