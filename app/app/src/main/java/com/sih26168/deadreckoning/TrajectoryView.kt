package com.sih26168.deadreckoning

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View

/**
 * Live trajectory trail, colored by FusionMode — same visual language as
 * results/blackout_demo_*.png (src/simulate_blackout.py's plots): blue for
 * GNSS-tracked, red for blackout (INS-only), orange for the reconnect
 * blend, plus an optional green map-matched overlay (MapMatcher's
 * road-snapped estimate) — the on-device version of the same
 * before/after comparison src/evaluate_with_mapmatching.py reports
 * offline. A simple auto-scaling scatter, not a real map — no tile/API-key
 * dependency needed for a demo.
 */
class TrajectoryView(context: Context, attrs: AttributeSet? = null) : View(context, attrs) {

    private data class Point(val x: Float, val y: Float, val mode: FusionMode)

    private val points = ArrayList<Point>()
    private val matchedPoints = ArrayList<FloatArray>()  // [x, y] pairs, same coordinate frame as points
    private var minX = 0f; private var maxX = 0f
    private var minY = 0f; private var maxY = 0f
    private var hasBounds = false

    private val paintGnss = Paint().apply { color = Color.parseColor("#2563eb"); style = Paint.Style.FILL; isAntiAlias = true }
    private val paintBlackout = Paint().apply { color = Color.parseColor("#dc2626"); style = Paint.Style.FILL; isAntiAlias = true }
    private val paintBlend = Paint().apply { color = Color.parseColor("#f59e0b"); style = Paint.Style.FILL; isAntiAlias = true }
    private val paintMatched = Paint().apply { color = Color.parseColor("#16a34a"); style = Paint.Style.STROKE; strokeWidth = 3f; isAntiAlias = true }
    private val paintBg = Paint().apply { color = Color.parseColor("#f7fafc") }

    private fun growBounds(x: Float, y: Float) {
        if (!hasBounds) {
            minX = x; maxX = x; minY = y; maxY = y; hasBounds = true
        } else {
            if (x < minX) minX = x; if (x > maxX) maxX = x
            if (y < minY) minY = y; if (y > maxY) maxY = y
        }
    }

    fun addPoint(x: Float, y: Float, mode: FusionMode) {
        points.add(Point(x, y, mode))
        growBounds(x, y)
        if (points.size > 5000) points.removeAt(0)  // bound memory on a long-running demo
        invalidate()
    }

    /** Adds a map-matched (road-snapped) point, in the same local xy frame
     * as addPoint's — see FusionEngine.localXYToLatLon / MapMatcher. */
    fun addMatchedPoint(x: Float, y: Float) {
        matchedPoints.add(floatArrayOf(x, y))
        growBounds(x, y)
        if (matchedPoints.size > 5000) matchedPoints.removeAt(0)
        invalidate()
    }

    fun clear() {
        points.clear()
        matchedPoints.clear()
        hasBounds = false
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        canvas.drawRect(0f, 0f, width.toFloat(), height.toFloat(), paintBg)
        if (!hasBounds) return

        val pad = 40f
        val spanX = (maxX - minX).coerceAtLeast(1f)
        val spanY = (maxY - minY).coerceAtLeast(1f)
        val scale = minOf((width - 2 * pad) / spanX, (height - 2 * pad) / spanY)

        fun sx(x: Float) = pad + (x - minX) * scale
        fun sy(y: Float) = height - pad - (y - minY) * scale  // flip: north = up

        for (p in points) {
            val paint = when (p.mode) {
                FusionMode.GNSS_TRACKING -> paintGnss
                FusionMode.BLACKOUT -> paintBlackout
                FusionMode.BLEND -> paintBlend
            }
            canvas.drawCircle(sx(p.x), sy(p.y), 5f, paint)
        }
        for (m in matchedPoints) {
            canvas.drawCircle(sx(m[0]), sy(m[1]), 7f, paintMatched)
        }
    }
}
