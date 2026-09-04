package com.idp.deviceintegrity

import android.os.Bundle
import android.widget.ScrollView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

/**
 * MainActivity
 *
 * Minimal demo screen for Layer 1. On launch, runs the full device
 * integrity check and prints a human-readable verdict to the screen.
 *
 * This is intentionally simple — its only purpose is to give a live,
 * demonstrable output for Review 2 ("show running code, partial
 * results and at least one measured metric"). The measured metric
 * here is processingTimeMs, shown alongside the verdict.
 */
class MainActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val textView = TextView(this).apply {
            textSize = 14f
            setPadding(32, 32, 32, 32)
        }
        val scrollView = ScrollView(this).apply {
            addView(textView)
        }
        setContentView(scrollView)

        val checker = DeviceIntegrityChecker(applicationContext)
        val verdict = checker.runIntegrityCheck()

        textView.text = formatVerdict(verdict)
    }

    private fun formatVerdict(verdict: Layer1Verdict): String {
        val sb = StringBuilder()
        sb.appendLine("=== Layer 1: Device Integrity Check ===\n")
        sb.appendLine("Overall decision: ${verdict.decision}")
        sb.appendLine("Processing time: ${verdict.processingTimeMs} ms\n")

        sb.appendLine("--- Root / Jailbreak Detection ---")
        sb.appendLine("Flagged: ${verdict.rootResult.flagged}")
        sb.appendLine("Confidence: ${verdict.rootResult.confidence}")
        sb.appendLine("Signals: ${verdict.rootResult.signals.ifEmpty { listOf("none") }}\n")

        sb.appendLine("--- Emulator Detection ---")
        sb.appendLine("Flagged: ${verdict.emulatorResult.flagged}")
        sb.appendLine("Confidence: ${verdict.emulatorResult.confidence}")
        sb.appendLine("Signals: ${verdict.emulatorResult.signals.ifEmpty { listOf("none") }}\n")

        sb.appendLine("--- Virtual Camera Detection ---")
        sb.appendLine("Flagged: ${verdict.virtualCameraResult.flagged}")
        sb.appendLine("Confidence: ${verdict.virtualCameraResult.confidence}")
        sb.appendLine("Signals: ${verdict.virtualCameraResult.signals.ifEmpty { listOf("none") }}")

        return sb.toString()
    }
}
