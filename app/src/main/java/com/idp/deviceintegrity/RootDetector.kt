package com.idp.deviceintegrity

import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import java.io.File

/**
 * RootDetector
 *
 * Layer 1a — Root / Jailbreak Detection
 *
 * Determines whether the Android device has had its OS-level security
 * restrictions removed (rooted). A rooted device grants elevated
 * privileges that attackers exploit to install SDK hooking tools,
 * modify system binaries, or run injection-attack frameworks that
 * feed synthetic (deepfake) video into the camera pipeline.
 *
 * This detector uses multiple independent heuristics rather than a
 * single check, since any one signal can be individually spoofed or
 * hidden by root-cloaking tools (e.g. Magisk Hide). Combining several
 * weak signals produces a stronger overall verdict.
 */
class RootDetector(private val context: Context) {

    // Common binary paths where the 'su' (superuser) executable is
    // installed by rooting tools. Presence of 'su' means an app can
    // request elevated privileges — something a stock, unrooted
    // Android device never allows.
    private val suPaths = arrayOf(
        "/system/app/Superuser.apk",
        "/sbin/su",
        "/system/bin/su",
        "/system/xbin/su",
        "/data/local/xbin/su",
        "/data/local/bin/su",
        "/system/sd/xbin/su",
        "/system/bin/failsafe/su",
        "/data/local/su",
        "/su/bin/su",
        "/system/xbin/daemonsu"
    )

    // Package names of well-known root management / root-hiding apps.
    private val rootPackages = arrayOf(
        "com.noshufou.android.su",
        "com.noshufou.android.su.elite",
        "eu.chainfire.supersu",
        "com.koushikdutta.superuser",
        "com.thirdparty.superuser",
        "com.yellowes.su",
        "com.topjohnwu.magisk",
        "com.kingroot.kinguser",
        "com.kingo.root",
        "com.smedialink.oneclickroot",
        "com.zhiqupk.root.global",
        "com.alephzain.framaroot"
    )

    /**
     * Runs all root-detection heuristics and returns an aggregated result.
     */
    fun detect(): DetectionResult {
        val signals = mutableListOf<String>()

        if (checkSuBinaryExists()) signals.add("su_binary_present")
        if (checkRootManagementApps()) signals.add("root_app_installed")
        if (checkTestKeysBuildTag()) signals.add("test_keys_build_tag")
        if (checkSystemWritable()) signals.add("system_partition_writable")
        if (checkRwPaths()) signals.add("rw_paths_on_ro_partitions")
        if (checkSuInPath()) signals.add("su_in_path_env")

        val isRooted = signals.isNotEmpty()
        // Confidence scales with number of independent signals found,
        // capped at 1.0. A single weak signal (e.g. test-keys tag, which
        // some legitimate custom ROMs also carry) gets partial confidence;
        // multiple corroborating signals push confidence to near-certain.
        val confidence = (signals.size / 3.0).coerceAtMost(1.0)

        return DetectionResult(
            flagged = isRooted,
            confidence = confidence,
            signals = signals
        )
    }

    private fun checkSuBinaryExists(): Boolean {
        return suPaths.any { path ->
            try {
                File(path).exists()
            } catch (e: Exception) {
                false
            }
        }
    }

    private fun checkRootManagementApps(): Boolean {
        val pm: PackageManager = context.packageManager
        return rootPackages.any { pkg ->
            try {
                pm.getPackageInfo(pkg, 0)
                true
            } catch (e: PackageManager.NameNotFoundException) {
                false
            }
        }
    }

    /**
     * Stock, production Android builds are signed with 'release-keys'.
     * Custom or debug builds (common on rooted/modified devices) are
     * signed with 'test-keys'. This alone is weak evidence (some OEMs
     * ship legitimate test-keys builds), so it only contributes one
     * signal toward the overall confidence score.
     */
    private fun checkTestKeysBuildTag(): Boolean {
        val tags = Build.TAGS
        return tags != null && tags.contains("test-keys")
    }

    /**
     * On a stock device, /system is mounted read-only. Rooting tools
     * frequently remount it read-write to patch system binaries.
     */
    private fun checkSystemWritable(): Boolean {
        return try {
            val systemDir = File("/system")
            systemDir.exists() && systemDir.canWrite()
        } catch (e: Exception) {
            false
        }
    }

    private fun checkRwPaths(): Boolean {
        val paths = arrayOf("/system", "/system/bin", "/system/sbin", "/system/xbin", "/vendor/bin")
        return paths.any { path ->
            try {
                val dir = File(path)
                dir.exists() && dir.canWrite()
            } catch (e: Exception) {
                false
            }
        }
    }

    private fun checkSuInPath(): Boolean {
        return try {
            val pathEnv = System.getenv("PATH") ?: return false
            pathEnv.split(":").any { dir ->
                File(dir, "su").exists()
            }
        } catch (e: Exception) {
            false
        }
    }
}
