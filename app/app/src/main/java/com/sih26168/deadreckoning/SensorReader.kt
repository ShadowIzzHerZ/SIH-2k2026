package com.sih26168.deadreckoning

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager

/**
 * Wraps SensorManager for raw accelerometer + gyroscope, always exposing
 * the most recent sample of each — the app's tick loop (MainActivity)
 * samples these at 10Hz to match src/data/windowing.py's
 * sample_rate_hz, not whatever raw rate the OS delivers events at.
 */
class SensorReader(context: Context) : SensorEventListener {
    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val accelSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
    private val gyroSensor = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

    // Volatile: written on the sensor thread, read from the tick loop.
    @Volatile var lastAccel: FloatArray = floatArrayOf(0f, 0f, 9.81f)
        private set
    @Volatile var lastGyro: FloatArray = floatArrayOf(0f, 0f, 0f)
        private set

    val available: Boolean get() = accelSensor != null && gyroSensor != null

    fun start() {
        // GAME delay (~50Hz) comfortably oversamples the 10Hz the model
        // expects; the tick loop reads whatever's most recent each tick
        // rather than needing every individual event.
        accelSensor?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }
        gyroSensor?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }
    }

    fun stop() {
        sensorManager.unregisterListener(this)
    }

    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> lastAccel = event.values.copyOf()
            Sensor.TYPE_GYROSCOPE -> lastGyro = event.values.copyOf()
        }
    }

    override fun onAccuracyChanged(sensor: Sensor, accuracy: Int) {}
}
