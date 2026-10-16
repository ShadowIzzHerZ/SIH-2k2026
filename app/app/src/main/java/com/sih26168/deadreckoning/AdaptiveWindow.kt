package com.sih26168.deadreckoning

import android.app.Activity
import android.view.View
import androidx.core.view.ViewCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat

/**
 * Screen-size / system-bar adaptation shared by every activity, so layouts
 * don't depend on one phone's status bar, gesture bar, cutout or width.
 */
object AdaptiveWindow {
    /** Widest the content column is allowed to get, in dp. Past this
     * (tablets, foldables) the extra width becomes side margin. */
    private const val MAX_CONTENT_DP = 600

    /** Draw edge to edge and add each system-bar inset as padding on the
     * given views, on top of whatever padding they already declare. */
    fun applyInsets(activity: Activity, root: View, top: View? = null, bottom: View? = null) {
        WindowCompat.setDecorFitsSystemWindows(activity.window, false)
        val topBase = top?.paddingTop ?: 0
        val bottomBase = (bottom ?: root).paddingBottom
        ViewCompat.setOnApplyWindowInsetsListener(root) { _, insets ->
            val bars = insets.getInsets(
                WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout()
            )
            if (top != null) {
                top.setPadding(top.paddingLeft, topBase + bars.top, top.paddingRight, top.paddingBottom)
            } else {
                root.setPadding(root.paddingLeft, bars.top, root.paddingRight, root.paddingBottom)
            }
            (bottom ?: root).setPadding(
                (bottom ?: root).paddingLeft, (bottom ?: root).paddingTop,
                (bottom ?: root).paddingRight, bottomBase + bars.bottom
            )
            insets
        }
        ViewCompat.requestApplyInsets(root)
    }

    /** Light status/navigation bar icons, since every screen is light. */
    fun lightBars(activity: Activity, root: View) {
        val c = WindowCompat.getInsetsController(activity.window, root)
        c.isAppearanceLightStatusBars = true
        c.isAppearanceLightNavigationBars = true
    }

    /** Centers a scrolling form's content column and caps its width. */
    fun capContentWidth(activity: Activity, content: View) {
        val dm = activity.resources.displayMetrics
        val widthDp = dm.widthPixels / dm.density
        if (widthDp <= MAX_CONTENT_DP) return
        val extra = ((widthDp - MAX_CONTENT_DP) / 2f * dm.density).toInt()
        content.setPadding(
            content.paddingLeft + extra, content.paddingTop,
            content.paddingRight + extra, content.paddingBottom
        )
    }
}
