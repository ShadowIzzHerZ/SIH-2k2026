package com.sih26168.deadreckoning

import android.content.Context
import android.util.AttributeSet
import android.widget.ScrollView

/** ScrollView that never grows past a fraction of the screen height, so
 * expanded floating cards scroll instead of overflowing small screens. A
 * smaller height limit from the parent still applies. */
class MaxHeightScrollView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : ScrollView(context, attrs) {
    override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) {
        var cap = (resources.displayMetrics.heightPixels * 0.5f).toInt()
        // Replacing the parent's spec outright let this grow past the room
        // the parent actually has (e.g. a short landscape or split-screen
        // window), pushing the rest of the card off screen.
        if (MeasureSpec.getMode(heightMeasureSpec) != MeasureSpec.UNSPECIFIED) {
            cap = minOf(cap, MeasureSpec.getSize(heightMeasureSpec))
        }
        super.onMeasure(widthMeasureSpec, MeasureSpec.makeMeasureSpec(cap, MeasureSpec.AT_MOST))
    }
}
