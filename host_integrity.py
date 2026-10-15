"""
Host & Camera Integrity Attestation Engine (Layer 1)
====================================================
Answers, on the local machine, "is the frame source a physical camera sensor in an
untampered execution environment?" before any pixel analysis runs.

Passive checks (no camera access, ~10 ms):
  - V4L2 node inspection: VIDIOC_QUERYCAP per-node device_caps (a metadata node is
    not a camera), driver / card / bus_info, sysfs placement of the node
    (/sys/devices/virtual => software device), bound kernel driver, USB/PCI parent
    and the physical USB identity (VID:PID, serial, port path).
  - Loaded kernel modules that create software cameras (v4l2loopback, akvcam, vivid ...).
  - Running injection tooling, matched on exact process names / command lines
    (OBS, DroidCam, ManyCam, DeepFaceLive, pyvirtualcam, ffmpeg/gstreamer -> v4l2).
  - Execution environment: hypervisor CPU flag / DMI vendor, containers,
    ptrace tracer, LD_PRELOAD, effective root.

Active check (needs the open capture stream, ~1 s):
  - Randomised multi-control challenge-response. Brightness (falling back to gamma /
    gain) and two controls drawn at random from contrast, saturation, hue and
    white-balance temperature are driven at once through VIDIOC_S_CTRL on the attested
    node, each with its own random balanced +/- sequence, while frames are read from the
    capture stream. Frames produced by that physical camera show every control in its
    own image feature; frames from a loopback device, a hooked capture call, a replay,
    or a replay that only follows the brightness flash do not.

Gating (CEN/TS 18099 style):
  BLOCK            debugger / LD_PRELOAD hook, VM or container, root, selected node
                   is not a physical capture device, or the active challenge fails
  FLAG_FOR_REVIEW  selected camera is physical, but software-camera modules or
                   injection tools are present on the host
  PASS             otherwise
"""

from __future__ import annotations

import fcntl
import glob
import json
import os
import random
import re
import struct
import subprocess
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# V4L2 ABI constants (linux/videodev2.h)
# --------------------------------------------------------------------------- #

VIDIOC_QUERYCAP = 0x80685600      # _IOR('V', 0, struct v4l2_capability)  (104 bytes)
VIDIOC_G_CTRL = 0xC008561B        # _IOWR('V', 27, struct v4l2_control)   (8 bytes)
VIDIOC_S_CTRL = 0xC008561C        # _IOWR('V', 28, struct v4l2_control)
VIDIOC_QUERYCTRL = 0xC0445624     # _IOWR('V', 36, struct v4l2_queryctrl) (68 bytes)

V4L2_CAP_VIDEO_CAPTURE = 0x00000001
V4L2_CAP_VIDEO_OUTPUT = 0x00000002
V4L2_CAP_VIDEO_CAPTURE_MPLANE = 0x00001000
V4L2_CAP_META_CAPTURE = 0x00800000
V4L2_CAP_STREAMING = 0x04000000
V4L2_CAP_DEVICE_CAPS = 0x80000000

V4L2_CTRL_FLAG_DISABLED = 0x0001
V4L2_CTRL_FLAG_READ_ONLY = 0x0004
V4L2_CTRL_FLAG_INACTIVE = 0x0010
V4L2_CTRL_FLAG_NEXT_CTRL = 0x80000000
V4L2_CTRL_TYPE_INTEGER = 1
V4L2_CTRL_TYPE_BOOLEAN = 2

V4L2_CID_BRIGHTNESS = 0x00980900
V4L2_CID_CONTRAST = 0x00980901
V4L2_CID_SATURATION = 0x00980902
V4L2_CID_HUE = 0x00980903
V4L2_CID_AUTO_WHITE_BALANCE = 0x0098090C
V4L2_CID_GAMMA = 0x00980910
V4L2_CID_GAIN = 0x00980913
V4L2_CID_WHITE_BALANCE_TEMPERATURE = 0x0098091A
CHALLENGE_CONTROLS = [("brightness", V4L2_CID_BRIGHTNESS), ("gamma", V4L2_CID_GAMMA), ("gain", V4L2_CID_GAIN)]

# --------------------------------------------------------------------------- #
# Signatures
# --------------------------------------------------------------------------- #

# Kernel drivers (QUERYCAP.driver / sysfs driver name) that implement software cameras.
VIRTUAL_CAMERA_DRIVERS = {
    "v4l2 loopback", "v4l2loopback", "akvcam", "vivid", "vimc", "vcam", "obs-virtualcam",
}
# Kernel modules that create software cameras when loaded.
VIRTUAL_CAMERA_MODULES = {"v4l2loopback", "akvcam", "vivid", "vimc", "vcam"}
# Exact process names (/proc/<pid>/comm, max 15 chars) of camera injection tools.
INJECTION_PROCESS_NAMES = {
    "obs", "obs64", "obs-studio", "droidcam", "droidcam-cli", "manycam", "deepfacelive",
    "snapcamera", "xsplit.vcam", "camtwist", "webcamoid",
}
# Command-line patterns (lower-cased) that indicate frames being written into a V4L2 device.
INJECTION_CMDLINE_PATTERNS = [
    re.compile(r"deepfacelive"),
    re.compile(r"pyvirtualcam"),
    re.compile(r"(^|\s)-f\s+v4l2(\s|$).*?/dev/video\d+"),
    re.compile(r"v4l2sink"),
]

HYPERVISOR_SIGNATURES = [
    "qemu", "virtualbox", "vmware", "bochs", "kvm", "xen", "bhyve", "innotek",
    "parallels", "hyper-v", "microsoft corporation virtual",
]

# Android-specific su locations. /usr/bin/su exists on every desktop Linux and is not an indicator.
ANDROID_SU_PATHS = ["/system/bin/su", "/system/xbin/su", "/system/sbin/su", "/vendor/bin/su", "/su/bin/su"]

CHASSIS_TYPES = {
    "1": "Other", "2": "Unknown", "3": "Desktop", "4": "Low Profile Desktop", "8": "Portable",
    "9": "Laptop", "10": "Notebook", "11": "Hand Held", "13": "All in One", "14": "Sub Notebook",
    "30": "Tablet", "31": "Convertible", "32": "Detachable",
}


def _read(path: str) -> Optional[str]:
    try:
        with open(path, "r", errors="ignore") as f:
            return f.read().strip()
    except OSError:
        return None


def _cstr(buf: bytes) -> str:
    return buf.split(b"\x00")[0].decode("utf-8", errors="ignore").strip()


# --------------------------------------------------------------------------- #
# V4L2 ioctls
# --------------------------------------------------------------------------- #

def _query_v4l2_ioctl(device_node: str) -> Dict[str, Any]:
    """VIDIOC_QUERYCAP. Uses the per-node device_caps when the driver reports them."""
    info = {"driver": "", "card": "", "bus_info": "", "caps": 0, "device_caps": 0,
            "is_capture": False, "is_output": False, "is_metadata": False, "is_streaming": False,
            "error": None}
    try:
        fd = os.open(device_node, os.O_RDONLY | os.O_NONBLOCK)
    except OSError as e:
        info["error"] = f"open failed: {e.strerror}"
        return info
    try:
        buf = bytearray(104)
        fcntl.ioctl(fd, VIDIOC_QUERYCAP, buf, True)
    except OSError as e:
        info["error"] = f"VIDIOC_QUERYCAP failed: {e.strerror}"
        return info
    finally:
        os.close(fd)
    caps, device_caps = struct.unpack_from("<II", buf, 84)
    eff = device_caps if caps & V4L2_CAP_DEVICE_CAPS else caps
    info.update({
        "driver": _cstr(buf[0:16]),
        "card": _cstr(buf[16:48]),
        "bus_info": _cstr(buf[48:80]),
        "caps": caps,
        "device_caps": eff,
        "is_capture": bool(eff & (V4L2_CAP_VIDEO_CAPTURE | V4L2_CAP_VIDEO_CAPTURE_MPLANE)),
        "is_output": bool(eff & V4L2_CAP_VIDEO_OUTPUT),
        "is_metadata": bool(eff & V4L2_CAP_META_CAPTURE),
        "is_streaming": bool(eff & V4L2_CAP_STREAMING),
    })
    return info


def query_controls(device_node: str) -> Dict[str, Dict[str, int]]:
    """Enumerates user/camera controls via VIDIOC_QUERYCTRL | V4L2_CTRL_FLAG_NEXT_CTRL."""
    controls: Dict[str, Dict[str, int]] = {}
    try:
        fd = os.open(device_node, os.O_RDONLY | os.O_NONBLOCK)
    except OSError:
        return controls
    try:
        cid = V4L2_CTRL_FLAG_NEXT_CTRL
        for _ in range(256):
            buf = bytearray(68)
            struct.pack_into("<I", buf, 0, cid)
            try:
                fcntl.ioctl(fd, VIDIOC_QUERYCTRL, buf, True)
            except OSError:
                break
            qid, qtype = struct.unpack_from("<II", buf, 0)
            mn, mx, step, default, flags = struct.unpack_from("<iiiiI", buf, 40)
            controls[_cstr(buf[8:40]) or hex(qid)] = {
                "id": qid, "type": qtype, "min": mn, "max": mx, "step": step,
                "default": default, "flags": flags,
            }
            cid = qid | V4L2_CTRL_FLAG_NEXT_CTRL
    finally:
        os.close(fd)
    return controls


def _get_ctrl(fd: int, cid: int) -> int:
    buf = bytearray(struct.pack("<Ii", cid, 0))
    fcntl.ioctl(fd, VIDIOC_G_CTRL, buf, True)
    return struct.unpack("<Ii", buf)[1]


def _set_ctrl(fd: int, cid: int, value: int) -> None:
    buf = bytearray(struct.pack("<Ii", cid, int(value)))
    fcntl.ioctl(fd, VIDIOC_S_CTRL, buf, True)


# --------------------------------------------------------------------------- #
# Device enumeration & classification
# --------------------------------------------------------------------------- #

def _usb_identity(device_path: str) -> Optional[Dict[str, str]]:
    """Walks up from a sysfs device path to the USB device directory (the one with idVendor)."""
    p = device_path
    while p and p != "/" and "/usb" in p:
        if os.path.exists(os.path.join(p, "idVendor")):
            return {
                "vid": _read(os.path.join(p, "idVendor")) or "",
                "pid": _read(os.path.join(p, "idProduct")) or "",
                "manufacturer": _read(os.path.join(p, "manufacturer")) or "",
                "product": _read(os.path.join(p, "product")) or "",
                "serial": _read(os.path.join(p, "serial")) or "",
                "port_path": os.path.basename(p),
                "speed_mbps": _read(os.path.join(p, "speed")) or "",
            }
        p = os.path.dirname(p)
    return None


def _pci_identity(device_path: str) -> Optional[Dict[str, str]]:
    p = device_path
    while p and p != "/":
        if os.path.exists(os.path.join(p, "vendor")) and os.path.exists(os.path.join(p, "class")) and "/pci" in p:
            return {"vendor": _read(os.path.join(p, "vendor")) or "", "device": _read(os.path.join(p, "device")) or "",
                    "slot": os.path.basename(p)}
        p = os.path.dirname(p)
    return None


def classify_video_node(sysfs_entry: str, dev_root: str = "/dev", ioctl_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Classifies one /sys/class/video4linux/videoN entry. `ioctl_info` can be injected for tests.
    A node is a *physical camera* only if it is a video-capture node, it is not placed under
    /sys/devices/virtual, it has a USB or PCI parent device, and no layer reports a known
    software-camera driver.
    """
    name = os.path.basename(sysfs_entry)
    node = os.path.join(dev_root, name)
    card_name = _read(os.path.join(sysfs_entry, "name")) or ""
    node_sysfs = os.path.realpath(sysfs_entry)
    dev_link = os.path.join(sysfs_entry, "device")
    parent = os.path.realpath(dev_link) if os.path.lexists(dev_link) else ""
    drv_link = os.path.join(dev_link, "driver")
    kernel_driver = os.path.basename(os.path.realpath(drv_link)) if os.path.lexists(drv_link) else ""

    if parent and "/usb" in parent:
        bus = "usb"
    elif parent and "/devices/pci" in parent:
        bus = "pci"
    elif parent and "/devices/platform" in parent:
        bus = "platform"
    else:
        bus = "none"

    q = ioctl_info if ioctl_info is not None else _query_v4l2_ioctl(node)
    drv = (q.get("driver") or "").lower()
    bus_info = (q.get("bus_info") or "").lower()

    reasons: List[str] = []
    if "/devices/virtual/" in node_sysfs:
        reasons.append("node registered under /sys/devices/virtual (no hardware parent)")
    if not parent:
        reasons.append("no parent hardware device in sysfs")
    if drv in VIRTUAL_CAMERA_DRIVERS or kernel_driver.lower() in VIRTUAL_CAMERA_DRIVERS:
        reasons.append(f"software camera driver '{q.get('driver') or kernel_driver}'")
    if any(sig in bus_info for sig in ("loopback", "akvcam", "vivid", "vimc")):
        reasons.append(f"software bus_info '{q.get('bus_info')}'")
    if q.get("is_output") and q.get("is_capture"):
        reasons.append("node accepts frames from user space (VIDEO_OUTPUT + VIDEO_CAPTURE)")

    is_virtual = bool(reasons)
    is_capture = bool(q.get("is_capture"))
    is_physical = (not is_virtual) and is_capture and bus in ("usb", "pci")

    usb = _usb_identity(parent) if bus == "usb" else None
    pci = _pci_identity(parent) if bus == "pci" else None
    if usb:
        camera_id = f"usb-{usb['vid']}-{usb['pid']}-{usb['serial'] or usb['port_path']}"
    elif pci:
        camera_id = f"pci-{pci['vendor']}-{pci['device']}-{pci['slot']}"
    else:
        camera_id = f"{bus}-{name}"

    return {
        "node": node,
        "name": card_name or q.get("card") or name,
        "card": q.get("card") or card_name,
        "driver": q.get("driver") or kernel_driver or "unknown",
        "kernel_driver": kernel_driver,
        "bus": bus,
        "bus_info": q.get("bus_info") or "",
        "bus_sysfs": parent,
        "node_sysfs": node_sysfs,
        "sysfs_path": sysfs_entry,
        "is_capture": is_capture,
        "is_metadata": bool(q.get("is_metadata")),
        "is_streaming": bool(q.get("is_streaming")),
        "device_caps": q.get("device_caps", 0),
        "is_virtual": is_virtual,
        "is_physical": is_physical,
        "virtual_reasons": reasons,
        "usb": usb,
        "pci": pci,
        "camera_id": re.sub(r"[^A-Za-z0-9._-]", "_", camera_id),
        "ioctl_error": q.get("error"),
    }


def enumerate_video_devices(sysfs_root: str = "/sys", dev_root: str = "/dev", capture_only: bool = False) -> List[Dict[str, Any]]:
    """Enumerates and classifies every V4L2 node on the host."""
    entries = sorted(
        glob.glob(os.path.join(sysfs_root, "class/video4linux/video*")),
        key=lambda p: int(re.sub(r"\D", "", os.path.basename(p)) or 0),
    )
    devices = [classify_video_node(e, dev_root=dev_root) for e in entries]
    if capture_only:
        devices = [d for d in devices if d["is_capture"]]
    return devices


def scan_loaded_modules(proc_root: str = "/proc") -> List[str]:
    found = []
    text = _read(os.path.join(proc_root, "modules")) or ""
    for line in text.splitlines():
        mod = line.split(" ", 1)[0].lower()
        if mod in VIRTUAL_CAMERA_MODULES:
            found.append(mod)
    return found


def scan_injection_processes(proc_root: str = "/proc", exclude_pids: Sequence[int] = ()) -> List[str]:
    """Exact-name match on comm, regex match on cmdline. Unreadable processes are skipped."""
    hits = []
    try:
        pids = [p for p in os.listdir(proc_root) if p.isdigit()]
    except OSError:
        return hits
    for pid in pids:
        if int(pid) in exclude_pids:
            continue
        comm = (_read(os.path.join(proc_root, pid, "comm")) or "").lower()
        if comm in INJECTION_PROCESS_NAMES:
            hits.append(f"{comm} (PID {pid})")
            continue
        try:
            with open(os.path.join(proc_root, pid, "cmdline"), "rb") as f:
                cmd = f.read(4096).replace(b"\x00", b" ").decode("utf-8", errors="ignore").lower()
        except OSError:
            continue
        if any(p.search(cmd) for p in INJECTION_CMDLINE_PATTERNS):
            hits.append(f"{comm or '?'} (PID {pid}): {cmd[:80].strip()}")
    return hits


def inspect_camera_hardware(device_node: Optional[str] = None, sysfs_root: str = "/sys",
                            dev_root: str = "/dev", proc_root: str = "/proc") -> Dict[str, Any]:
    """Audits the selected capture node plus host-wide software-camera indicators."""
    devices = enumerate_video_devices(sysfs_root, dev_root)
    capture_devices = [d for d in devices if d["is_capture"]]
    if device_node is None:
        physical = [d for d in capture_devices if d["is_physical"]]
        target = (physical or capture_devices or devices or [{"node": os.path.join(dev_root, "video0")}])[0]["node"]
    else:
        target = device_node
    selected = next((d for d in devices if d["node"] == target), None)

    modules = scan_loaded_modules(proc_root)
    procs = scan_injection_processes(proc_root, exclude_pids=(os.getpid(),))
    other_virtual = [d["node"] for d in devices if d["is_virtual"] and d["node"] != target]

    return {
        "target_node": target,
        "selected_device": selected,
        "all_devices": devices,
        "found": selected is not None,
        "is_capture": bool(selected and selected["is_capture"]),
        "is_physical": bool(selected and selected["is_physical"]),
        "is_virtual": bool(selected and selected["is_virtual"]),
        "camera_id": selected["camera_id"] if selected else None,
        "primary_card": selected["card"] if selected else "None",
        "primary_driver": selected["driver"] if selected else "None",
        "primary_bus": selected["bus_info"] if selected else "None",
        "sysfs_device_tree": selected["bus_sysfs"] if selected else "",
        "loopback_modules": modules,
        "injection_processes": procs,
        "other_virtual_nodes": other_virtual,
    }


# --------------------------------------------------------------------------- #
# Platform & runtime
# --------------------------------------------------------------------------- #

def inspect_platform_and_hardware(sysfs_root: str = "/sys", proc_root: str = "/proc") -> Dict[str, Any]:
    """Hypervisor / container detection plus descriptive platform telemetry."""
    vm_reasons: List[str] = []

    dmi: Dict[str, str] = {}
    for key in ["sys_vendor", "product_name", "product_version", "bios_vendor", "bios_version", "chassis_type", "board_vendor"]:
        val = _read(os.path.join(sysfs_root, "class/dmi/id", key))
        if val is not None:
            dmi[key] = val
            low = val.lower()
            sig = next((s for s in HYPERVISOR_SIGNATURES if re.search(rf"\b{re.escape(s)}\b", low)), None)
            if sig and key != "chassis_type":
                vm_reasons.append(f"DMI {key} = '{val}'")

    hyp_type = _read(os.path.join(sysfs_root, "hypervisor/type"))
    if hyp_type:
        vm_reasons.append(f"/sys/hypervisor/type = {hyp_type}")

    cpu_model, threads, hypervisor_flag = "Unknown CPU", 0, False
    cpuinfo = _read(os.path.join(proc_root, "cpuinfo")) or ""
    for line in cpuinfo.splitlines():
        key = line.split(":", 1)[0].strip()
        if key == "processor":
            threads += 1
        elif key == "model name" and cpu_model == "Unknown CPU":
            cpu_model = line.split(":", 1)[1].strip()
        elif key == "flags" and not hypervisor_flag:
            hypervisor_flag = "hypervisor" in line.split(":", 1)[1].split()
    if hypervisor_flag:
        vm_reasons.append("CPUID hypervisor bit set")

    container_reasons = []
    if os.path.exists("/.dockerenv"):
        container_reasons.append("/.dockerenv present")
    if os.path.exists("/run/.containerenv"):
        container_reasons.append("/run/.containerenv present")
    cgroup = _read(os.path.join(proc_root, "1/cgroup")) or ""
    if re.search(r"(docker|kubepods|containerd|lxc)", cgroup):
        container_reasons.append("PID 1 cgroup is a container cgroup")

    battery = None
    bats = sorted(glob.glob(os.path.join(sysfs_root, "class/power_supply/BAT*")))
    if bats:
        b = bats[0]
        parts = [_read(os.path.join(b, k)) or "" for k in ("manufacturer", "model_name")]
        battery = f"{' '.join(p for p in parts if p)} ({_read(os.path.join(b, 'capacity')) or '?'}%, {_read(os.path.join(b, 'status')) or '?'})".strip()

    thermal = None
    zones = sorted(glob.glob(os.path.join(sysfs_root, "class/thermal/thermal_zone*/temp")))
    if zones:
        try:
            thermal = round(float(_read(zones[0]) or "nan") / 1000.0, 1)
        except ValueError:
            thermal = None

    is_vm = bool(vm_reasons)
    is_container = bool(container_reasons)
    return {
        "is_bare_metal": not (is_vm or is_container),
        "is_vm": is_vm,
        "is_container": is_container,
        "vm_reasons": vm_reasons,
        "container_reasons": container_reasons,
        "vendor": dmi.get("sys_vendor", "unknown"),
        "product": dmi.get("product_version") or dmi.get("product_name", "unknown"),
        "bios": f"{dmi.get('bios_vendor', '')} {dmi.get('bios_version', '')}".strip() or "unknown",
        "chassis": CHASSIS_TYPES.get(dmi.get("chassis_type", ""), "unknown"),
        "cpu_model": f"{cpu_model} ({threads} threads)",
        "hypervisor_cpu_flag": hypervisor_flag,
        "battery_detected": battery is not None,
        "battery_summary": battery or "none",
        "thermal_temp_c": thermal,
        "dmi_raw": dmi,
    }


def inspect_runtime_anti_tampering(proc_root: str = "/proc") -> Dict[str, Any]:
    """ptrace tracer, LD_PRELOAD injection, effective UID, effective capabilities."""
    tracer_pid, cap_eff = 0, "unknown"
    status = _read(os.path.join(proc_root, "self/status")) or ""
    for line in status.splitlines():
        if line.startswith("TracerPid:"):
            tracer_pid = int(line.split(":")[1].strip() or 0)
        elif line.startswith("CapEff:"):
            cap_eff = line.split(":")[1].strip()
    ld_preload = os.environ.get("LD_PRELOAD")
    preload_file = _read("/etc/ld.so.preload")
    uid = os.geteuid() if hasattr(os, "geteuid") else -1
    su_found = [p for p in ANDROID_SU_PATHS if os.path.exists(p)]
    return {
        "tracer_pid": tracer_pid,
        "is_debugger_attached": tracer_pid > 0,
        "ld_preload": ld_preload,
        "ld_so_preload": preload_file or None,
        "is_injected": bool((ld_preload and ld_preload.strip()) or (preload_file and preload_file.strip())),
        "uid": uid,
        "is_root": uid == 0,
        "cap_eff": cap_eff,
        "su_present": bool(su_found),
        "su_paths": su_found,
    }


def probe_hypervisor() -> Dict[str, Any]:
    plat = inspect_platform_and_hardware()
    return {"is_vm": plat["is_vm"], "is_container": plat["is_container"], "vendor": plat["vendor"]}


def probe_privileges() -> Dict[str, Any]:
    t = inspect_runtime_anti_tampering()
    return {"is_root": t["is_root"], "su_present": t["su_present"], "su_paths": t["su_paths"], "uid": t["uid"]}


# --------------------------------------------------------------------------- #
# Passive host attestation (Gate 1a)
# --------------------------------------------------------------------------- #

def probe_host_integrity(target_camera_node: Optional[str] = None, sysfs_root: str = "/sys",
                         dev_root: str = "/dev", proc_root: str = "/proc") -> Dict[str, Any]:
    """Runs all passive Layer 1 checks against the camera node that will be captured from."""
    t0 = time.perf_counter()
    cam = inspect_camera_hardware(target_camera_node, sysfs_root, dev_root, proc_root)
    plat = inspect_platform_and_hardware(sysfs_root, proc_root)
    tamp = inspect_runtime_anti_tampering(proc_root)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    emulator = plat["is_vm"] or plat["is_container"]
    debugger = tamp["is_debugger_attached"] or tamp["is_injected"]
    root = tamp["is_root"]
    sel = cam["selected_device"]
    camera_not_physical = not cam["is_physical"]
    host_context_flags = bool(cam["loopback_modules"] or cam["injection_processes"] or cam["other_virtual_nodes"])

    if debugger:
        verdict, reason = "BLOCK", (f"Runtime hooking detected (TracerPid={tamp['tracer_pid']}, "
                                    f"LD_PRELOAD={tamp['ld_preload'] or tamp['ld_so_preload']}).")
    elif emulator:
        verdict, reason = "BLOCK", f"Virtualised execution environment: {'; '.join(plat['vm_reasons'] + plat['container_reasons'])}."
    elif root:
        verdict, reason = "BLOCK", "Capture process runs with effective UID 0 (root)."
    elif camera_not_physical:
        if not cam["found"]:
            why = f"capture node {cam['target_node']} does not exist"
        elif sel["is_virtual"]:
            why = f"{sel['node']} is a software camera: {'; '.join(sel['virtual_reasons'])}"
        elif not sel["is_capture"]:
            why = f"{sel['node']} is not a video-capture node (device_caps=0x{sel['device_caps']:08x})"
        else:
            why = f"{sel['node']} has no USB/PCI hardware parent (bus={sel['bus']})"
        verdict, reason = "BLOCK", f"Frame source is not a physical camera: {why}."
    elif host_context_flags:
        bits = []
        if cam["loopback_modules"]:
            bits.append(f"software-camera kernel modules loaded: {', '.join(cam['loopback_modules'])}")
        if cam["other_virtual_nodes"]:
            bits.append(f"other software camera nodes: {', '.join(cam['other_virtual_nodes'])}")
        if cam["injection_processes"]:
            bits.append(f"injection tooling running: {', '.join(cam['injection_processes'])}")
        verdict, reason = "FLAG_FOR_REVIEW", "Selected camera is physical, but " + "; ".join(bits) + "."
    else:
        verdict, reason = "PASS", (f"Physical camera {sel['card']} ({sel['driver']}, {sel['bus_info']}) on "
                                   f"{plat['vendor']} {plat['product']}; no hooks, virtualisation or software cameras.")

    def st(ok: bool, bad: str = "BLOCK") -> str:
        return "PASS" if ok else bad

    diagnostics = [
        {"name": "Capture node is a physical camera", "status": st(not camera_not_physical),
         "value": (f"{sel['node']}: {sel['card']} ({sel['driver']} on {sel['bus_info'] or sel['bus']})" if sel else cam["target_node"]),
         "details": ("; ".join(sel["virtual_reasons"]) if sel and sel["virtual_reasons"]
                     else (f"USB {sel['usb']['vid']}:{sel['usb']['pid']} {sel['usb']['product']} port {sel['usb']['port_path']}"
                           if sel and sel.get("usb") else "")) or "-"},
        {"name": "Software-camera modules / nodes", "status": st(not (cam["loopback_modules"] or cam["other_virtual_nodes"]), "WARN"),
         "value": ", ".join(cam["loopback_modules"] + cam["other_virtual_nodes"]) or "none", "details": "/proc/modules, sysfs"},
        {"name": "Injection tooling processes", "status": st(not cam["injection_processes"], "WARN"),
         "value": ", ".join(cam["injection_processes"]) or "none", "details": "exact comm / cmdline match"},
        {"name": "Bare-metal execution", "status": st(not emulator),
         "value": f"{plat['vendor']} {plat['product']} ({plat['chassis']})",
         "details": "; ".join(plat["vm_reasons"] + plat["container_reasons"]) or "no hypervisor bit, DMI or container markers"},
        {"name": "Debugger / linker hooks", "status": st(not debugger),
         "value": f"TracerPid={tamp['tracer_pid']}, LD_PRELOAD={tamp['ld_preload'] or 'unset'}", "details": "/proc/self/status, environ, /etc/ld.so.preload"},
        {"name": "Privilege", "status": st(not root),
         "value": f"EUID={tamp['uid']}, CapEff={tamp['cap_eff']}", "details": "capture should run unprivileged"},
        {"name": "Platform telemetry (informational)", "status": "INFO",
         "value": f"CPU {plat['cpu_model']}; battery {plat['battery_summary']}; thermal {plat['thermal_temp_c']} C",
         "details": "descriptive only; not used for the verdict"},
    ]

    return {
        "passed": verdict != "BLOCK",
        "blocked": verdict == "BLOCK",
        "verdict": verdict,
        "details": reason,
        "root_detected": root,
        "emulator_detected": emulator,
        "virtual_camera_detected": cam["is_virtual"] or bool(cam["loopback_modules"]),
        "debugger_detected": debugger,
        "camera_id": cam["camera_id"],
        "devices": cam["all_devices"],
        "camera_audit": cam,
        "hardware_telemetry": plat,
        "anti_tampering": tamp,
        "diagnostics_summary": diagnostics,
        "vm_info": {"is_vm": plat["is_vm"], "is_container": plat["is_container"], "vendor": plat["vendor"]},
        "privileges": {"is_root": root, "su_present": tamp["su_present"], "su_paths": tamp["su_paths"], "uid": tamp["uid"]},
        "latency_ms": round(latency_ms, 2),
    }


# --------------------------------------------------------------------------- #
# Active sensor challenge (Gate 1b)
# --------------------------------------------------------------------------- #

CHALLENGE_CONFIG = {
    "slots": 12,             # number of +/- slots (balanced per control)
    "frames_per_slot": 4,
    "min_sign_changes": 5,
    "delta_fraction": 0.1,   # brightness amplitude as a fraction of the control's range
    "extra_controls": 2,     # controls drawn at random besides brightness
    "max_sequence_corr": 0.34,  # max |correlation| between two controls' sequences
    "min_chroma": 2.0,       # scene chroma needed before saturation / hue can be drawn
    "max_lag": 3,            # frames of pipeline latency tolerated
    "min_corr": 0.85,        # partial correlation (feature vs own command | trend, other commands)
    "min_effect": 6.0,       # grey levels between + and - brightness slots required
    "attempts": 2,           # a FAIL needs every attempt to fail
}

# Controls the challenge can drive, the image feature each one moves (see challenge_features),
# its amplitude as a fraction of the control's range and the smallest response accepted.
# `signed`: the direction is known (brighter, more contrast, more saturated); hue rotation and
# white-balance direction differ between cameras, so those are judged on magnitude.
CHALLENGE_CHANNELS = {
    "brightness": {"cid": V4L2_CID_BRIGHTNESS, "feature": "luma", "delta_fraction": 0.10,
                   "min_effect": 6.0, "signed": True},
    "contrast": {"cid": V4L2_CID_CONTRAST, "feature": "spread", "delta_fraction": 0.04,
                 "min_effect": 0.08, "signed": True},
    "saturation": {"cid": V4L2_CID_SATURATION, "feature": "chroma", "delta_fraction": 0.25,
                   "min_effect": 0.08, "signed": True, "needs_colour": True},
    "hue": {"cid": V4L2_CID_HUE, "feature": "hue", "delta_fraction": 0.10,
            "min_effect": 4.0, "signed": False, "needs_colour": True},
    "white_balance": {"cid": V4L2_CID_WHITE_BALANCE_TEMPERATURE, "feature": "wb", "delta_fraction": 0.20,
                      "min_effect": 0.04, "signed": False, "needs_manual_wb": True},
}


def make_challenge_sequence(slots: int, min_sign_changes: int, rng: Optional[random.Random] = None) -> List[int]:
    rng = rng or random.SystemRandom()
    base = [1] * (slots // 2) + [-1] * (slots - slots // 2)
    while True:
        seq = base[:]
        rng.shuffle(seq)
        if sum(1 for a, b in zip(seq, seq[1:]) if a != b) >= min_sign_changes:
            return seq


def analyze_challenge_response(luma: Sequence[float], command: Sequence[int], max_lag: int = 3,
                               min_corr: float = 0.9, min_effect: float = 6.0) -> Dict[str, Any]:
    """
    Partial correlation between per-frame mean luma and the commanded +/-1 sequence,
    controlling for a linear trend (auto-exposure drift), over 0..max_lag frames of latency.
    Both series are residualised on [1, t] before correlating, so the trend fit cannot
    absorb the commanded square wave.
    """
    l = np.asarray(luma, dtype=np.float64)
    c = np.asarray(command, dtype=np.float64)
    if len(l) < 8 or np.std(l) < 1e-9:
        return {"passed": False, "corr": 0.0, "effect": 0.0, "lag": 0}
    best = {"corr": -1.0, "effect": 0.0, "lag": 0}
    for lag in range(0, max_lag + 1):
        cc = c[: len(c) - lag] if lag else c
        ll = l[lag:]
        if np.std(cc) < 1e-9 or np.std(ll) < 1e-9:
            continue
        design = np.column_stack([np.ones(len(ll)), np.arange(len(ll), dtype=np.float64)])
        proj = design @ np.linalg.pinv(design)
        rl, rc = ll - proj @ ll, cc - proj @ cc
        if np.std(rl) < 1e-9 or np.std(rc) < 1e-9:
            continue
        r = float(np.dot(rl, rc) / (np.linalg.norm(rl) * np.linalg.norm(rc)))
        if r > best["corr"]:
            best = {"corr": r, "effect": float(ll[cc > 0].mean() - ll[cc < 0].mean()), "lag": lag}
    best["passed"] = best["corr"] >= min_corr and best["effect"] >= min_effect
    best["corr"] = round(best["corr"], 3)
    best["effect"] = round(best["effect"], 2)
    return best


def _small_bgr(frame: np.ndarray) -> np.ndarray:
    f = frame if frame.ndim == 3 else np.repeat(frame[..., None], 3, axis=2)
    return cv2.resize(f, (80, 60), interpolation=cv2.INTER_AREA).astype(np.float32)


def _small_gray(frame: np.ndarray) -> np.ndarray:
    g = frame if frame.ndim == 2 else frame.mean(axis=2)
    return cv2.resize(g.astype(np.float32), (80, 60), interpolation=cv2.INTER_AREA)


def brightness_shift(frame: np.ndarray, reference: np.ndarray) -> float:
    """
    Global brightness change of `frame` relative to `reference` (an 80x60 grey thumbnail):
    the median per-pixel difference. A brightness command shifts every pixel equally, while
    a person moving in front of the camera changes only part of the image, so the median
    stays on the commanded shift as long as less than half of the frame moves.
    """
    return float(np.median(_small_gray(frame) - reference))


def _image_stats(frame: np.ndarray) -> Dict[str, Any]:
    s = _small_bgr(frame)
    b, g, r = s[..., 0], s[..., 1], s[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    c = (0.564 * (b - y)) + 1j * (0.713 * (r - y))      # Cb + i Cr
    p10, p90 = np.percentile(y, (10, 90))
    return {"y": y, "c": c, "spread": max(float(p90 - p10), 1.0),
            "chroma": max(float(np.mean(np.abs(c))), 0.1), "wb": float(np.log((b.mean() + 1) / (r.mean() + 1)))}


def challenge_features(frame: np.ndarray, ref: Dict[str, Any]) -> Dict[str, float]:
    """
    One value per challenge feature, relative to the reference frame's statistics `ref`:
      luma    median grey-level shift (brightness / gamma / gain)
      spread  log ratio of the 10-90 % luma range (contrast)
      chroma  log ratio of the mean chroma magnitude (saturation)
      hue     rotation of the chroma plane in degrees, pixel-wise against the reference (hue)
      wb      change of log(mean blue / mean red) (white-balance temperature)
    """
    st = _image_stats(frame)
    rot = np.sum(st["c"] * np.conj(ref["c"]))
    return {"luma": float(np.median(st["y"] - ref["y"])),
            "spread": float(np.log(st["spread"] / ref["spread"])),
            "chroma": float(np.log(st["chroma"] / ref["chroma"])),
            "hue": float(np.degrees(np.angle(rot))) if abs(rot) > 1e-6 else 0.0,
            "wb": st["wb"] - ref["wb"]}


def make_independent_sequences(n: int, slots: int, min_sign_changes: int, max_corr: float,
                               rng: Optional[random.Random] = None) -> List[List[int]]:
    """`n` balanced +/-1 sequences whose pairwise |correlation| is <= max_corr, so the
    response to each control can be separated from the others by regression."""
    rng = rng or random.SystemRandom()
    for _ in range(1000):
        seqs: List[List[int]] = []
        for _ in range(200):
            s = make_challenge_sequence(slots, min_sign_changes, rng)
            if all(abs(float(np.dot(s, o))) / slots <= max_corr for o in seqs):
                seqs.append(s)
                if len(seqs) == n:
                    return seqs
    raise RuntimeError("could not draw independent challenge sequences")


def analyze_multi_response(features: Dict[str, Sequence[float]], commands: Dict[str, Sequence[int]],
                           channels: Dict[str, Dict[str, Any]], max_lag: int = 3,
                           min_corr: float = 0.85) -> Dict[str, Dict[str, Any]]:
    """
    For every driven control, the partial correlation between its image feature and its own
    +/-1 command, controlling for a linear trend (auto-exposure drift) and for the commands of
    the other controls driven at the same time, over 0..max_lag frames of latency. `effect` is
    the regression estimate of the feature change between the + and - levels. Controls whose
    direction depends on the camera (hue, white balance) are judged on |r|.
    """
    names = list(commands)
    out: Dict[str, Dict[str, Any]] = {}
    for name in names:
        spec = channels[name]
        f = np.asarray(features[spec["feature"]], dtype=np.float64)
        best = {"corr": -1.0, "effect": 0.0, "lag": 0}
        for lag in range(0, max_lag + 1):
            n = len(f) - lag
            if n < 8:
                continue
            y = f[lag:]
            cmd = {k: np.asarray(commands[k][:n], dtype=np.float64) for k in names}
            x = cmd[name]
            z = np.column_stack([np.ones(n), np.arange(n, dtype=np.float64)] + [cmd[k] for k in names if k != name])
            proj = z @ np.linalg.pinv(z)
            ry, rx = y - proj @ y, x - proj @ x
            if np.linalg.norm(ry) < 1e-9 or np.linalg.norm(rx) < 1e-9:
                continue
            r = float(np.dot(ry, rx) / (np.linalg.norm(ry) * np.linalg.norm(rx)))
            effect = 2.0 * float(np.dot(ry, rx) / np.dot(rx, rx))  # + level minus - level
            if not spec["signed"]:
                r, effect = abs(r), abs(effect)
            if r > best["corr"]:
                best = {"corr": r, "effect": effect, "lag": lag}
        best["passed"] = bool(best["corr"] >= min_corr and best["effect"] >= spec["min_effect"])
        best["corr"] = round(best["corr"], 3)
        best["effect"] = round(best["effect"], 3)
        out[name] = best
    return out


def _writable(ctl: Optional[Dict[str, int]]) -> bool:
    return bool(ctl) and ctl["type"] == V4L2_CTRL_TYPE_INTEGER and not ctl["flags"] & (
        V4L2_CTRL_FLAG_DISABLED | V4L2_CTRL_FLAG_READ_ONLY)


def run_sensor_challenge(device_node: str, read_frame: Callable[[], Optional[np.ndarray]],
                         cfg: Optional[dict] = None, on_frame: Optional[Callable[[np.ndarray, int, int], None]] = None,
                         rng: Optional[random.Random] = None) -> Dict[str, Any]:
    """
    Randomised multi-control challenge. Brightness (or gamma / gain when the camera has no
    brightness control) plus `extra_controls` controls drawn at random from contrast,
    saturation, hue and white-balance temperature are driven at the same time through
    VIDIOC_S_CTRL on `device_node`, each with its own random balanced +/- sequence, while
    frames are read from the capture stream under test with `read_frame`. Every control
    must show up in its own image feature (luma, luma range, chroma, chroma angle, blue/red
    balance), so a stream that only follows a brightness flash still fails. The original
    control values (and automatic white balance) are restored afterwards.
    Returns PASS / FAIL / UNSUPPORTED.

    `read_frame` may return a frame, or (frame, capture_timestamp_ms) with the driver's
    CLOCK_MONOTONIC timestamp (cv2.CAP_PROP_POS_MSEC on V4L2). With timestamps, each frame is
    labelled with the commands active at capture time, which removes the variable latency of
    the driver's frame queue (it grows at low frame rates).

    `on_challenge_frame`, if given in cfg, receives every frame read during the challenge:
    those frames are attested to come from this sensor and serve as the PRNU reference for
    the same-sensor check.
    """
    cfg = {**CHALLENGE_CONFIG, **(cfg or {})}
    rng = rng or random.SystemRandom()
    keep = cfg.get("on_challenge_frame")
    t0 = time.perf_counter()
    controls = {v["id"]: v for v in query_controls(device_node).values()}
    primary = next(((n, cid) for n, cid in CHALLENGE_CONTROLS if _writable(controls.get(cid))
                    and not controls[cid]["flags"] & V4L2_CTRL_FLAG_INACTIVE), None)
    if primary is None:
        return {"verdict": "UNSUPPORTED", "passed": None, "control": None, "controls": [],
                "controls_available": sorted(str(v) for v in controls),
                "details": "No writable brightness/gamma/gain control on this node; challenge skipped.",
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 1)}
    has_awb = _writable(controls.get(V4L2_CID_AUTO_WHITE_BALANCE)) or (
        controls.get(V4L2_CID_AUTO_WHITE_BALANCE, {}).get("type") == V4L2_CTRL_TYPE_BOOLEAN)
    extras = [n for n, spec in CHALLENGE_CHANNELS.items() if n != "brightness" and _writable(controls.get(spec["cid"]))
              and (not controls[spec["cid"]]["flags"] & V4L2_CTRL_FLAG_INACTIVE or (spec.get("needs_manual_wb") and has_awb))]

    try:
        fd = os.open(device_node, os.O_RDWR | os.O_NONBLOCK)
    except OSError as e:
        return {"verdict": "UNSUPPORTED", "passed": None, "control": primary[0], "controls": [primary[0]],
                "details": f"Cannot open {device_node} for control access: {e.strerror}", "latency_ms": 0.0}

    def read():
        got = read_frame()
        frame, ts = got if isinstance(got, tuple) else (got, None)
        if frame is not None and keep:
            keep(frame)
        return frame, ts

    originals: Dict[int, int] = {}
    attempts = []
    total = cfg["attempts"] * cfg["slots"] * cfg["frames_per_slot"]
    k = 0
    try:
        for attempt_i in range(cfg["attempts"]):
            frame, _ = read()
            if frame is None:
                break
            ref = _image_stats(frame)
            # Chroma controls are invisible in a nearly grey scene, so they are only drawn when
            # the scene has colour to rotate or saturate.
            pool = [n for n in extras if not CHALLENGE_CHANNELS[n].get("needs_colour") or ref["chroma"] >= cfg["min_chroma"]]
            chosen = [primary[0]] + rng.sample(pool, min(cfg["extra_controls"], len(pool)))
            specs = {primary[0]: {**CHALLENGE_CHANNELS["brightness"], "cid": primary[1]}}
            specs.update({n: CHALLENGE_CHANNELS[n] for n in chosen[1:]})
            if any(s.get("needs_manual_wb") for s in specs.values()) and V4L2_CID_AUTO_WHITE_BALANCE not in originals:
                originals[V4L2_CID_AUTO_WHITE_BALANCE] = _get_ctrl(fd, V4L2_CID_AUTO_WHITE_BALANCE)
                _set_ctrl(fd, V4L2_CID_AUTO_WHITE_BALANCE, 0)
                for _ in range(3):  # let the colour step from leaving auto white balance pass
                    read()
            levels = {}
            for n, s in list(specs.items()):
                ctl = controls[s["cid"]]
                if s["cid"] not in originals:
                    originals[s["cid"]] = _get_ctrl(fd, s["cid"])
                base, step = originals[s["cid"]], max(1, ctl["step"])
                delta = max(step, int(round(s["delta_fraction"] * (ctl["max"] - ctl["min"]) / step)) * step)
                lo, hi = max(ctl["min"], base - delta), min(ctl["max"], base + delta)
                if hi - lo < 2 * step:
                    if n == primary[0]:
                        return {"verdict": "UNSUPPORTED", "passed": None, "control": n, "controls": [n],
                                "details": f"Control {n} has no usable range around {base}.", "latency_ms": 0.0}
                    del specs[n]
                    continue
                levels[n] = (lo, hi)
            names = list(specs)
            seqs = dict(zip(names, make_independent_sequences(len(names), cfg["slots"], cfg["min_sign_changes"],
                                                                 cfg["max_sequence_corr"], rng)))
            frame, _ = read()
            if frame is None:
                break
            ref = _image_stats(frame)
            feats: Dict[str, List[float]] = {f: [] for f in ("luma", "spread", "chroma", "hue", "wb")}
            command: Dict[str, List[int]] = {n: [] for n in names}
            set_times: List[tuple] = []  # (monotonic ms when the slot's commands were applied, slot index)
            for i in range(cfg["slots"]):
                for n in names:
                    _set_ctrl(fd, specs[n]["cid"], levels[n][1] if seqs[n][i] > 0 else levels[n][0])
                set_times.append((time.monotonic() * 1000.0, i))
                for _ in range(cfg["frames_per_slot"]):
                    frame, ts = read()
                    if frame is None:
                        break
                    k += 1
                    if on_frame:
                        on_frame(frame, k, total)
                    if ts is not None:
                        # Label the frame with the commands in effect when the sensor captured it,
                        # so frames that waited in the driver queue are not mislabelled.
                        active = [j for (t_set, j) in set_times if t_set <= ts]
                        if not active:
                            continue  # captured before the challenge started
                        slot = active[-1]
                    else:
                        slot = i
                    for f, v in challenge_features(frame, ref).items():
                        feats[f].append(v)
                    for n in names:
                        command[n].append(seqs[n][slot])
            for n in names:
                _set_ctrl(fd, specs[n]["cid"], originals[specs[n]["cid"]])
            per = analyze_multi_response(feats, command, specs, cfg["max_lag"], cfg["min_corr"])
            p = per[primary[0]]
            attempts.append({"passed": all(c["passed"] for c in per.values()), "controls": names,
                             "channels": {n: {**per[n], "levels": list(levels[n])} for n in names},
                             "corr": min(c["corr"] for c in per.values()), "effect": p["effect"], "lag": p["lag"],
                             "sequence": seqs[primary[0]], "levels": list(levels[primary[0]]),
                             "frames": len(feats["luma"]), "luma": [round(v, 2) for v in feats["luma"]],
                             "command": command[primary[0]]})
            if attempts[-1]["passed"]:
                break
            for _ in range(3):  # let the sensor settle before a retry
                read()
    except OSError as e:
        return {"verdict": "UNSUPPORTED", "passed": None, "control": primary[0], "controls": [primary[0]],
                "details": f"VIDIOC_S_CTRL failed during the challenge: {e.strerror}", "latency_ms": 0.0}
    finally:
        # Manual white-balance temperature first, then automatic white balance back on.
        for cid in sorted(originals, key=lambda c: c == V4L2_CID_AUTO_WHITE_BALANCE):
            try:
                _set_ctrl(fd, cid, originals[cid])
            except OSError:
                pass
        os.close(fd)

    if not attempts:
        return {"verdict": "UNSUPPORTED", "passed": None, "control": primary[0], "controls": [primary[0]],
                "details": "The stream delivered no frames during the challenge.",
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 1)}
    passed = any(a["passed"] for a in attempts)
    last = next((a for a in attempts if a["passed"]), attempts[-1])
    names = last["controls"]
    missed = [n for n in names if not last["channels"][n]["passed"]]
    summary = ", ".join(f"{n} r={last['channels'][n]['corr']:.2f}" for n in names)
    return {
        "verdict": "PASS" if passed else "FAIL",
        "passed": passed,
        "control": " + ".join(names),
        "controls": names,
        "channels": last["channels"],
        "original_value": originals.get(CHALLENGE_CHANNELS["brightness"]["cid"] if primary[0] == "brightness"
                                        else primary[1]),
        "levels": last["levels"],
        "corr": last["corr"],
        "effect": last["effect"],
        "lag_frames": last["lag"],
        "attempts": attempts,
        "details": (f"Frames tracked a random {len(names)}-control challenge on {device_node} ({summary})."
                    if passed else
                    f"Frames did NOT follow the random {len(names)}-control challenge applied to {device_node} "
                    f"(no response to {', '.join(missed)}; {summary}): the stream is not produced by this sensor."),
        "latency_ms": round((time.perf_counter() - t0) * 1000.0, 1),
    }


def capture_attested_frames(device_node: str, n_frames: int, width: int = 640, height: int = 480,
                            fourcc: str = "MJPG", warmup_frames: int = 15, challenge: bool = True,
                            on_frame: Optional[Callable[[np.ndarray, int, int, str], None]] = None) -> Dict[str, Any]:
    """
    Opens `device_node` (V4L2 backend, by path so the attested node is the captured node),
    lets auto-exposure settle, runs the active sensor challenge on the same stream, then
    captures `n_frames` raw frames for Layers 2 and 3. Frames are never re-encoded.
    The frames read during the challenge are returned as `challenge_frames` (the attested
    PRNU reference for the same-sensor check).
    """
    import cv2

    out: Dict[str, Any] = {"frames": [], "challenge": None, "challenge_frames": [], "error": None, "node": device_node}
    cap = cv2.VideoCapture(device_node, cv2.CAP_V4L2)
    if not cap.isOpened():
        out["error"] = f"Could not open {device_node}"
        return out
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        def read() -> Optional[np.ndarray]:
            ok, f = cap.read()
            return f if ok else None

        for i in range(warmup_frames):
            f = read()
            if f is not None and on_frame:
                on_frame(f, i + 1, warmup_frames, "warmup")

        def read_ts():
            ok, f = cap.read()
            return (f, cap.get(cv2.CAP_PROP_POS_MSEC)) if ok else (None, None)

        if challenge:
            out["challenge"] = run_sensor_challenge(
                device_node, read_ts, cfg={"on_challenge_frame": out["challenge_frames"].append},
                on_frame=(lambda f, k, n: on_frame(f, k, n, "challenge")) if on_frame else None,
            )
            for _ in range(4):  # settle after restoring the control
                read()

        t0 = time.perf_counter()
        for i in range(n_frames):
            f = read()
            if f is None:
                break
            out["frames"].append(f)
            if on_frame:
                on_frame(f, i + 1, n_frames, "capture")
        dt = time.perf_counter() - t0
        out["fps"] = round(len(out["frames"]) / dt, 2) if dt > 0 else 0.0
        out["fourcc"] = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode("ascii", errors="replace")
        if out["frames"]:
            out["height"], out["width"] = out["frames"][0].shape[:2]
    finally:
        cap.release()
    return out


# --------------------------------------------------------------------------- #
# Client attestation payloads & simulation presets
# --------------------------------------------------------------------------- #

SIMULATION_PRESETS = {
    "Clean Physical Device": dict(passed=True, blocked=False, root_detected=False, emulator_detected=False,
                                  virtual_camera_detected=False, verdict="PASS",
                                  details="Simulated: hardware verified clean."),
    "Compromised / Rooted Android (Magisk/SU)": dict(passed=False, blocked=True, root_detected=True, emulator_detected=False,
                                                     virtual_camera_detected=False, verdict="BLOCK",
                                                     details="Simulated: root binary / Magisk detected."),
    "Android Emulator (Goldfish/QEMU)": dict(passed=False, blocked=True, root_detected=False, emulator_detected=True,
                                             virtual_camera_detected=False, verdict="BLOCK",
                                             details="Simulated: AVD goldfish/ranchu markers detected."),
    "Virtual Camera Injection (OBS / Hooked Driver)": dict(passed=False, blocked=True, root_detected=False, emulator_detected=False,
                                                           virtual_camera_detected=True, verdict="BLOCK",
                                                           details="Simulated: selected capture device is a software camera."),
}


def evaluate_client_attestation(payload: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluates a JSON attestation payload from the Android SDK, or a named simulation preset.
    Any other string falls back to a live probe of this host.
    """
    t0 = time.perf_counter()
    if isinstance(payload, str) and payload.strip().startswith("{"):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as e:
            return {"passed": False, "blocked": True, "verdict": "BLOCK", "root_detected": False,
                    "emulator_detected": False, "virtual_camera_detected": False, "latency_ms": 0.0,
                    "details": f"Malformed attestation payload: {e}"}

    if isinstance(payload, dict):
        root = bool(payload.get("isRooted", payload.get("root_detected", False)))
        emulator = bool(payload.get("isEmulator", payload.get("emulator_detected", False)))
        vcam = bool(payload.get("isVirtualCamera", payload.get("virtual_camera_detected", False)))
        if root or emulator:
            verdict = "BLOCK"
        elif vcam:
            # Android SDK treats camera enumeration anomalies as corroborating only (see VirtualCameraDetector.kt).
            verdict = "FLAG_FOR_REVIEW"
        else:
            verdict = "PASS"
        return {"passed": verdict != "BLOCK", "blocked": verdict == "BLOCK", "verdict": verdict,
                "root_detected": root, "emulator_detected": emulator, "virtual_camera_detected": vcam,
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                "details": payload.get("details") or f"Client attestation payload evaluated: {verdict}."}

    if payload in SIMULATION_PRESETS:
        return {**SIMULATION_PRESETS[payload], "simulated": True,
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2)}

    return probe_host_integrity()


def probe_uploaded_file_provenance(video_path: str) -> Dict[str, Any]:
    """
    A file has no live device to attest. Container metadata is reported for context only
    (it is trivially editable), and the session is marked UNATTESTED_ORIGIN.
    """
    t0 = time.perf_counter()
    meta: Dict[str, Any] = {}
    try:
        res = subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", video_path],
                             capture_output=True, text=True, timeout=5.0)
        if res.returncode == 0:
            meta = json.loads(res.stdout)
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        pass
    tags = {k.lower(): v for k, v in (meta.get("format", {}).get("tags", {}) or {}).items()}
    encoder = tags.get("encoder", "")
    device_tags = {k: v for k, v in tags.items() if any(s in k for s in ("make", "model", "com.apple", "com.android"))}
    software = any(s in encoder.lower() for s in ("lavf", "ffmpeg", "handbrake", "obs", "premiere", "davinci"))
    return {
        "passed": True,
        "blocked": False,
        "verdict": "UNATTESTED_ORIGIN",
        "root_detected": False,
        "emulator_detected": False,
        "virtual_camera_detected": False,
        "is_file_upload": True,
        "encoder": encoder or "not recorded",
        "is_software_encoder": software,
        "device_tags": device_tags,
        "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        "details": (f"Uploaded file: no live device to attest. Container encoder '{encoder or 'not recorded'}'"
                    f"{' (software re-encode)' if software else ''}"
                    f"{'; device tags ' + json.dumps(device_tags) if device_tags else ''}. Metadata is informational only."),
    }
