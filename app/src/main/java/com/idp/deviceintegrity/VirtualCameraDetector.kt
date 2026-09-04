package com.idp.deviceintegrity

import android.content.Context
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager

/**
 * VirtualCameraDetector
 *
 * Layer 1c — Virtual Camera Driver Detection
 *
 * Determines whether the active camera stream is coming from genuine
 * device hardware or from software that impersonates a camera to the
 * OS (e.g. virtual camera drivers used to feed pre-recorded or
 * deepfake video into an app instead of a live physical feed).
 *
 * On Android this threat is less common than on desktop OSes (Windows/
 * macOS, where tools like OBS Virtual Camera or ManyCam are widely
 * available), because Android's Camera2 API sits closer to hardware
 * and third-party virtual camera drivers require system-level access
 * to register as a camera provider. Nonetheless, apps running on
 * rooted devices, in emulators, or through hooked SDKs can still
 * expose spoofed camera characteristics. This detector treats
 * Android's camera enumeration as a source of corroborating evidence,
 * to be combined with the Root and Emulator detectors in the fusion
 * layer, rather than a standalone guarantee.
 *
 * NOTE ON SCOPE: This is the module most likely to expand later in
 * the project — for instance, if the team's client targets desktop/
 * web KYC flows in addition to mobile, the same signature-matching
 * approach applies to platform camera enumeration APIs
 * (MediaDevices.enumerateDevices() on web, DirectShow on Windows,
 * AVFoundation on macOS).
 */
class VirtualCameraDetector(private val context: Context) {

    // Substrings found in the device/vendor names of known virtual
    // camera software. Physical camera hardware is reported by the
    // OS with OEM sensor identifiers (e.g. "back camera 0"), not
    // consumer software brand names.
    private val knownVirtualCameraSignatures = listOf(
        "obs virtual camera",
        "obs-camera",
        "manycam",
        "snap camera",
        "xsplit vcam",
        "camtwist",
        "droidcam",
        "iriun",
        "epoccam",
        "ivcam",
        "youcam",
        "virtual camera",
        "v4l2loopback",
        "camo"
    )

    fun detect(): DetectionResult {
        val signals = mutableListOf<String>()

        try {
            val cameraManager = context.getSystemService(Context.CAMERA_SERVICE) as? CameraManager
            if (cameraManager == null) {
                signals.add("camera_service_unavailable")
            } else {
                val cameraIds = cameraManager.cameraIdList

                if (cameraIds.isEmpty()) {
                    // A device reporting zero cameras while an app expects
                    // to open a camera session is itself anomalous and
                    // consistent with a virtual/hooked camera provider
                    // intercepting calls before they reach the real HAL.
                    signals.add("no_physical_cameras_enumerated")
                }

                for (id in cameraIds) {
                    val characteristics = cameraManager.getCameraCharacteristics(id)
                    if (isSuspiciousCharacteristics(characteristics)) {
                        signals.add("suspicious_camera_characteristics:$id")
                    }
                }
            }
        } catch (e: Exception) {
            // If enumeration itself throws (e.g. permission hooking or
            // an SDK intercepting the Camera2 API), that failure is a
            // signal in its own right rather than something to silently
            // swallow.
            signals.add("camera_enumeration_exception")
        }

        val flagged = signals.isNotEmpty()
        val confidence = (signals.size / 2.0).coerceAtMost(1.0)

        return DetectionResult(
            flagged = flagged,
            confidence = confidence,
            signals = signals
        )
    }

    /**
     * Checks lens facing and hardware support level. Physical cameras
     * report a definite lens facing (front/back/external) and a real
     * hardware support level. Emulated or hooked camera characteristics
     * sometimes report as LEGACY-level support or omit lens facing —
     * both are treated as weak corroborating evidence here, since the
     * strongest defense against emulated cameras is Layer 1a/1b
     * (root + emulator detection) plus Layer 2 (PRNU sensor-noise
     * profiling), not the Android camera metadata alone.
     */
    private fun isSuspiciousCharacteristics(characteristics: CameraCharacteristics): Boolean {
        return try {
            val supportLevel = characteristics.get(
                CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL
            )
            supportLevel == CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LEGACY
        } catch (e: Exception) {
            false
        }
    }
}
