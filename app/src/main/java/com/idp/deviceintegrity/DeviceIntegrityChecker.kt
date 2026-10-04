package com.idp.deviceintegrity

import android.content.Context
import android.os.SystemClock

/**
 * DeviceIntegrityChecker
 *
 * Entry point for Layer 1 of the project's five-layer detection
 * pipeline: Hardware and Device Attestation.
 *
 * This is the first checkpoint a video/KYC session passes through,
 * before any deepfake-specific analysis (PRNU sensor-noise profiling,
 * FFT temporal analysis, or rPPG liveness — Layers 2 through 4) is
 * attempted. It exists to catch attacks at the transport/environment
 * level: rooted or jailbroken devices, emulators, and virtual camera
 * drivers, all of which are prerequisites attackers commonly rely on
 * to inject a synthetic video stream into the application pipeline.
 *
 * Because these checks only inspect device/OS metadata (not video
 * frames), they are computationally cheap and can run in well under
 * 50ms, making them suitable as a hard gate: if a critical anomaly is
 * found here, the session is terminated immediately, and the far more
 * expensive Layers 2-4 are never invoked. This is the "gated inference
 * fusion" strategy described in the project design — Layer 1's role
 * is specifically to make that early-exit decision possible.
 */
class DeviceIntegrityChecker(private val context: Context) {

    private val rootDetector = RootDetector(context)
    private val emulatorDetector = EmulatorDetector(context)
    private val virtualCameraDetector = VirtualCameraDetector(context)

    /**
     * Runs all Layer 1 sub-checks and returns a single aggregated
     * verdict for the session.
     */
    fun runIntegrityCheck(): Layer1Verdict {
        val startTime = SystemClock.elapsedRealtime()

        val rootResult = rootDetector.detect()
        val emulatorResult = emulatorDetector.detect()
        val cameraResult = virtualCameraDetector.detect()

        val elapsedMs = SystemClock.elapsedRealtime() - startTime

        // Gating logic: root or emulator detection are treated as
        // critical/definitive anomalies (per the project's Layer 5
        // gated-fusion design, these alone are sufficient to terminate
        // a session, since a compromised OS environment undermines
        // every later layer's trustworthiness). Virtual camera signals
        // are corroborating evidence on Android specifically (see
        // VirtualCameraDetector docstring) and are weighted accordingly,
        // but do not independently trigger termination.
        val criticalAnomalyFound = rootResult.flagged || emulatorResult.flagged

        val verdict = when {
            criticalAnomalyFound -> Layer1Decision.BLOCK
            cameraResult.flagged -> Layer1Decision.FLAG_FOR_REVIEW
            else -> Layer1Decision.PASS
        }

        return Layer1Verdict(
            decision = verdict,
            rootResult = rootResult,
            emulatorResult = emulatorResult,
            virtualCameraResult = cameraResult,
            processingTimeMs = elapsedMs
        )
    }
}

/**
 * The three possible outcomes of Layer 1. FLAG_FOR_REVIEW exists as a
 * middle ground for weaker/ corroborating-only signals (see camera
 * detector notes) so the pipeline is not forced into a strict binary
 * pass/fail on evidence that is not yet definitive on its own.
 */
enum class Layer1Decision {
    PASS,
    FLAG_FOR_REVIEW,
    BLOCK
}

/**
 * Aggregated Layer 1 output, passed downstream to whatever module
 * implements Layer 5 (gated inference fusion) in the full pipeline.
 */
data class Layer1Verdict(
    val decision: Layer1Decision,
    val rootResult: DetectionResult,
    val emulatorResult: DetectionResult,
    val virtualCameraResult: DetectionResult,
    val processingTimeMs: Long
)
