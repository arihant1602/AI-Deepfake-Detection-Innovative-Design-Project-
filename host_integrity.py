"""
Host & Environment Integrity Attestation Engine (Layer 1)
==========================================================
Probes the host operating system, video capture devices, hypervisor markers,
privilege levels, and processes in real time to detect injection attacks,
virtual camera drivers, and virtualized execution environments.

Also provides parsing for client attestation payloads (e.g. from Aarya's Android SDK)
and simulation presets for regression testing.
"""

from __future__ import annotations
import glob
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Union

VIRTUAL_CAMERA_SIGNATURES = [
    "v4l2loopback",
    "obs",
    "obs-virtualcam",
    "dummy",
    "dummy-uvc",
    "akvcam",
    "droidcam",
    "manycam",
    "fake",
    "vloopback",
    "streamlabs",
    "ipcamera",
]

HYPERVISOR_SIGNATURES = [
    "qemu",
    "virtualbox",
    "vmware",
    "bochs",
    "kvm",
    "xen",
    "bhyve",
    "innotek",
    "wsl",
    "parallels",
    "hyper-v",
]


def enumerate_video_devices() -> List[Dict[str, Any]]:
    """Discovers and inspects all video capture devices on the Linux system."""
    devices: List[Dict[str, Any]] = []

    # Check sysfs video4linux nodes
    vpaths = sorted(glob.glob("/sys/class/video4linux/video*"))
    for vpath in vpaths:
        dev_node = "/dev/" + os.path.basename(vpath)
        name_file = os.path.join(vpath, "name")
        card_name = "Unknown Video Device"
        if os.path.exists(name_file):
            try:
                with open(name_file, "r") as f:
                    card_name = f.read().strip()
            except Exception:
                pass

        card_lower = card_name.lower()
        is_virtual = any(sig in card_lower for sig in VIRTUAL_CAMERA_SIGNATURES)

        devices.append({
            "node": dev_node,
            "name": card_name,
            "is_virtual": is_virtual,
            "sysfs_path": vpath,
        })

    # Fallback if sysfs is restricted: probe /dev/video*
    if not devices:
        for node in sorted(glob.glob("/dev/video*")):
            devices.append({
                "node": node,
                "name": "Generic Video Device",
                "is_virtual": False,
                "sysfs_path": "",
            })

    return devices


def probe_hypervisor() -> Dict[str, Any]:
    """Detects whether the environment is running inside a VM or hypervisor."""
    is_vm = False
    vendor_found = "Bare Metal"

    # Check DMI system identifiers
    dmi_files = ["sys_vendor", "product_name", "bios_vendor", "board_vendor"]
    for dmi in dmi_files:
        p = f"/sys/class/dmi/id/{dmi}"
        if os.path.exists(p):
            try:
                with open(p, "r") as f:
                    val = f.read().strip()
                    val_lower = val.lower()
                    for sig in HYPERVISOR_SIGNATURES:
                        if sig in val_lower:
                            is_vm = True
                            vendor_found = f"{dmi}: {val}"
                            break
            except Exception:
                pass
        if is_vm:
            break

    # Check CPU flags in /proc/cpuinfo
    if not is_vm and os.path.exists("/proc/cpuinfo"):
        try:
            with open("/proc/cpuinfo", "r") as f:
                content = f.read().lower()
                if "hypervisor" in content:
                    is_vm = True
                    vendor_found = "CPU hypervisor flag present"
        except Exception:
            pass

    # Check container markers
    is_container = os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv")

    return {
        "is_vm": is_vm,
        "is_container": is_container,
        "vendor": vendor_found,
    }


def probe_privileges() -> Dict[str, Any]:
    """Inspects process execution privileges and root binaries."""
    is_root = False
    try:
        is_root = (os.geteuid() == 0)
    except AttributeError:
        pass

    su_paths = [
        "/system/bin/su",
        "/system/xbin/su",
        "/sbin/su",
        "/usr/bin/su",
        "/usr/local/bin/su",
    ]
    su_found = [p for p in su_paths if os.path.exists(p)]

    return {
        "is_root": is_root,
        "su_present": len(su_found) > 0,
        "su_paths": su_found,
        "uid": getattr(os, "geteuid", lambda: -1)(),
    }


def probe_host_integrity() -> Dict[str, Any]:
    """
    Executes real-time live host integrity attestation:
    - Queries active video devices for virtual camera loopbacks
    - Inspects hypervisor and container execution flags
    - Evaluates process privileges and root state
    - Measures genuine latency with microsecond precision
    """
    t0 = time.perf_counter()

    devices = enumerate_video_devices()
    vm_info = probe_hypervisor()
    priv_info = probe_privileges()

    latency_ms = (time.perf_counter() - t0) * 1000.0

    virtual_camera_detected = any(d["is_virtual"] for d in devices)
    emulator_detected = vm_info["is_vm"] or vm_info["is_container"]
    root_detected = priv_info["is_root"] or (priv_info["su_present"] and priv_info["uid"] == 0)

    # CEN/TS 18099 Gating Rules:
    # Root / Emulator -> BLOCK
    # Virtual Camera -> FLAG_FOR_REVIEW
    # Clean Physical -> PASS
    if root_detected:
        verdict = "BLOCK"
        passed = False
        blocked = True
        reason = "Privileged root execution detected on host environment."
    elif emulator_detected:
        verdict = "BLOCK"
        passed = False
        blocked = True
        reason = f"Hypervisor / containerized environment detected ({vm_info['vendor']})."
    elif virtual_camera_detected:
        verdict = "FLAG_FOR_REVIEW"
        passed = True
        blocked = False
        reason = "Virtual camera loopback driver detected (v4l2loopback/OBS)."
    else:
        verdict = "PASS"
        passed = True
        blocked = False
        phys_count = len(devices)
        reason = f"Hardware verified clean. {phys_count} physical camera device(s) enumerated on bare metal."

    return {
        "passed": passed,
        "blocked": blocked,
        "root_detected": root_detected,
        "emulator_detected": emulator_detected,
        "virtual_camera_detected": virtual_camera_detected,
        "devices": devices,
        "vm_info": vm_info,
        "privileges": priv_info,
        "latency_ms": round(latency_ms, 2),
        "verdict": verdict,
        "details": reason,
    }


def evaluate_client_attestation(payload: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluates an attestation payload received from an Android client (Aarya's SDK)
    or parses test simulation strings.
    """
    t0 = time.perf_counter()

    # If payload is a dictionary from client JSON:
    if isinstance(payload, dict):
        root = bool(payload.get("isRooted", payload.get("root_detected", False)))
        emulator = bool(payload.get("isEmulator", payload.get("emulator_detected", False)))
        vcam = bool(payload.get("isVirtualCamera", payload.get("virtual_camera_detected", False)))
        details = payload.get("details", "")

        if root or emulator:
            verdict = "BLOCK"
            passed = False
            blocked = True
        elif vcam:
            verdict = "FLAG_FOR_REVIEW"
            passed = True
            blocked = False
        else:
            verdict = "PASS"
            passed = True
            blocked = False

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "passed": passed,
            "blocked": blocked,
            "root_detected": root,
            "emulator_detected": emulator,
            "virtual_camera_detected": vcam,
            "latency_ms": round(latency_ms, 2),
            "verdict": verdict,
            "details": details or f"Evaluated client attestation: verdict={verdict}",
        }

    # If payload is a JSON string
    if isinstance(payload, str) and payload.strip().startswith("{"):
        try:
            parsed = json.loads(payload)
            return evaluate_client_attestation(parsed)
        except Exception:
            pass

    # Simulation presets (for regression testing & demo scenarios)
    presets = {
        "Clean Physical Device": {
            "passed": True,
            "blocked": False,
            "root_detected": False,
            "emulator_detected": False,
            "virtual_camera_detected": False,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "verdict": "PASS",
            "details": "Hardware verified clean. No root binaries, hypervisor markers, or virtual cameras detected.",
        },
        "Compromised / Rooted Android (Magisk/SU)": {
            "passed": False,
            "blocked": True,
            "root_detected": True,
            "emulator_detected": False,
            "virtual_camera_detected": False,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "verdict": "BLOCK",
            "details": "Root binary detected (su / Magisk / writable system partition). Device environment untrusted.",
        },
        "Android Emulator (Goldfish/QEMU)": {
            "passed": False,
            "blocked": True,
            "root_detected": False,
            "emulator_detected": True,
            "virtual_camera_detected": False,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "verdict": "BLOCK",
            "details": "AVD hypervisor fingerprint detected (sdk_gphone / goldfish markers). High injection risk.",
        },
        "Virtual Camera Injection (OBS / Hooked Driver)": {
            "passed": True,
            "blocked": False,
            "root_detected": False,
            "emulator_detected": False,
            "virtual_camera_detected": True,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "verdict": "FLAG_FOR_REVIEW",
            "details": "Virtual camera enumeration anomaly detected. Flagged for secondary PRNU confirmation.",
        },
    }

    if payload in presets:
        return presets[payload]

    # Default fallback: run real live probe
    return probe_host_integrity()


def probe_uploaded_file_provenance(video_path: str) -> Dict[str, Any]:
    """
    Evaluates origin provenance for a standalone uploaded video file:
    - Inspects container format, streams, and encoder tags (e.g. FFmpeg/Lavf vs native hardware encoders).
    - Notes that without an active client attestation session, device environment is UNATTESTED.
    - Defers definitive physical authenticity verification to Gate 2 (PRNU) and Gate 3 (Temporal dynamics).
    """
    t0 = time.perf_counter()
    import subprocess
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", video_path
    ]
    encoder = "Unknown"
    is_software = False
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=2.0)
        if res.returncode == 0:
            meta = json.loads(res.stdout)
            tags = meta.get("format", {}).get("tags", {})
            encoder = tags.get("encoder", tags.get("compatible_brands", "Unknown"))
            is_software = any(s in str(encoder).lower() for s in ["lavf", "ffmpeg", "handbrake", "obs", "premiere", "python"])
    except Exception:
        pass

    latency_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "passed": True,
        "blocked": False,
        "root_detected": False,
        "emulator_detected": False,
        "virtual_camera_detected": is_software,
        "is_file_upload": True,
        "encoder": encoder,
        "is_software_encoder": is_software,
        "latency_ms": round(latency_ms, 2),
        "verdict": "UNATTESTED_ORIGIN" if not is_software else "FLAG_FOR_REVIEW",
        "details": (
            f"Standalone video upload: no live client attestation payload attached. "
            f"Container encoder: '{encoder}'. Defers to Gate 2 (PRNU sensor noise) and Gate 3 (Temporal forensics)."
        ),
    }
