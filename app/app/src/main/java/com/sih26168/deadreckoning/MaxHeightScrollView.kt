package com.sih26168.deadreckoning

import android.content.Context
import android.util.AttributeSet
import android.widget.ScrollView

/** ScrollView that never grows past a fraction of the screen height, so
 * expanded floating cards scroll instead of overflowing small screens. */
class MaxHeightScrollView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : ScrollView(context, attrs) {
    override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) {
        val cap = (resources.displayMetrics.heightPixels * 0.5f).toInt()
        super.onMeasure(widthMeasureSpec, MeasureSpec.makeMeasureSpec(cap, MeasureSpec.AT_MOST))
    }
}
