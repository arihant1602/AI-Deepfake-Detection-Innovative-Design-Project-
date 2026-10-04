package com.idp.deviceintegrity

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorManager
import android.os.Build

/**
 * EmulatorDetector
 *
 * Layer 1b — Emulator / Virtual Device Detection
 *
 * Determines whether the app is running inside an Android emulator
 * (e.g. Android Studio AVD, Genymotion, BlueStacks) rather than on
 * physical hardware. Attackers favor emulators over real phones
 * because emulators are far easier to instrument with injection
 * tooling, debuggers, and virtual camera feeds — none of which
 * require physically owning or modifying a real device.
 *
 * Detection relies on the fact that emulators leave characteristic
 * traces in build metadata and hardware identifiers that real,
 * commercially sold devices do not have.
 */
class EmulatorDetector(private val context: Context) {

    fun detect(): DetectionResult {
        val signals = mutableListOf<String>()

        if (checkBuildFingerprint()) signals.add("emulator_build_fingerprint")
        if (checkBuildModel()) signals.add("emulator_build_model")
        if (checkBuildManufacturer()) signals.add("emulator_manufacturer")
        if (checkBuildHardware()) signals.add("emulator_hardware_tag")
        if (checkBuildProduct()) signals.add("emulator_product_name")
        if (checkQemuFiles()) signals.add("qemu_driver_files_present")
        if (checkSensorAbsence()) signals.add("missing_expected_sensors")

        val isEmulator = signals.isNotEmpty()
        val confidence = (signals.size / 3.0).coerceAtMost(1.0)

        return DetectionResult(
            flagged = isEmulator,
            confidence = confidence,
            signals = signals
        )
    }

    private fun checkBuildFingerprint(): Boolean {
        val fingerprint = Build.FINGERPRINT ?: return false
        return fingerprint.startsWith("generic")
                || fingerprint.startsWith("unknown")
                || fingerprint.contains("google/sdk")
                || fingerprint.contains("emulator")
                || fingerprint.contains("test-keys")
    }

    private fun checkBuildModel(): Boolean {
        val model = Build.MODEL ?: return false
        val markers = listOf("google_sdk", "Emulator", "Android SDK built for x86", "sdk_gphone")
        return markers.any { model.contains(it, ignoreCase = true) }
    }

    private fun checkBuildManufacturer(): Boolean {
        val manufacturer = Build.MANUFACTURER ?: return false
        return manufacturer.equals("Genymotion", ignoreCase = true)
                || manufacturer.contains("unknown", ignoreCase = true)
    }

    private fun checkBuildHardware(): Boolean {
        val hardware = Build.HARDWARE ?: return false
        val markers = listOf("goldfish", "ranchu", "vbox86")
        return markers.any { hardware.contains(it, ignoreCase = true) }
    }

    private fun checkBuildProduct(): Boolean {
        val product = Build.PRODUCT ?: return false
        val markers = listOf("sdk", "google_sdk", "sdk_x86", "vbox86p", "emulator")
        return markers.any { product.contains(it, ignoreCase = true) }
    }

    /**
     * QEMU (the virtualization layer most Android emulators are built on)
     * exposes pipe files used for host-guest communication. These do not
     * exist on physical hardware.
     */
    private fun checkQemuFiles(): Boolean {
        val qemuPaths = arrayOf(
            "/dev/qemu_pipe",
            "/dev/socket/qemud",
            "/system/lib/libc_malloc_debug_qemu.so",
            "/sys/qemu_trace"
        )
        return qemuPaths.any { path ->
            try {
                java.io.File(path).exists()
            } catch (e: Exception) {
                false
            }
        }
    }

    /**
     * Real phones ship with a baseline set of physical sensors
     * (accelerometer, gyroscope, proximity, light). Many emulator
     * configurations omit some or all of these, or report implausible
     * static/zero readings. Absence of an accelerometer in particular
     * is a strong emulator signal, since virtually every commercial
     * Android phone since ~2012 includes one.
     */
    private fun checkSensorAbsence(): Boolean {
        return try {
            val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as? SensorManager
                ?: return false
            val hasAccelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER) != null
            val hasGyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE) != null
            !hasAccelerometer && !hasGyroscope
        } catch (e: Exception) {
            false
        }
    }
}
