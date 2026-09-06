package com.idp.deviceintegrity

/**
 * DetectionResult
 *
 * Standard output format shared by all three Layer 1 sub-detectors
 * (Root, Emulator, Virtual Camera). Keeping this uniform makes it
 * straightforward to feed into the Layer 5 gated fusion mechanism
 * described in the project's overall architecture, without each
 * detector needing bespoke handling.
 *
 * @param flagged   true if this detector found at least one anomaly signal
 * @param confidence   0.0-1.0 score; more corroborating signals = higher confidence
 * @param signals   human-readable list of the specific checks that fired,
 *                  kept for logging/debugging and for the eventual
 *                  research report's results section
 */
data class DetectionResult(
    val flagged: Boolean,
    val confidence: Double,
    val signals: List<String>
)
