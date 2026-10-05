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
  - Sensor control challenge-response. A random, balanced +/- sequence is applied
    to a hardware image control (brightness, falling back to gamma / gain) through
    VIDIOC_S_CTRL on the attested node while frames are read from the capture
    stream. Frames produced by that physical camera track the sequence; frames
    from a loopback device, a hooked capture call or a replay do not.

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

V4L2_CID_BRIGHTNESS = 0x00980900
V4L2_CID_GAMMA = 0x00980910
V4L2_CID_GAIN = 0x00980913
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
    "slots": 12,             # number of +/- slots (balanced)
    "frames_per_slot": 4,
    "min_sign_changes": 5,
    "delta_fraction": 0.1,   # amplitude as a fraction of the control's range
    "max_lag": 3,            # frames of pipeline latency tolerated
    "min_corr": 0.9,         # partial correlation (luma vs command | trend) required
    "min_effect": 6.0,       # grey levels between + and - slots required
    "attempts": 2,           # a FAIL needs every attempt to fail
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


def run_sensor_challenge(device_node: str, read_frame: Callable[[], Optional[np.ndarray]],
                         cfg: Optional[dict] = None, on_frame: Optional[Callable[[np.ndarray, int, int], None]] = None,
                         rng: Optional[random.Random] = None) -> Dict[str, Any]:
    """
    Applies a random balanced +/- sequence to a hardware image control of `device_node`
    (VIDIOC_S_CTRL) while reading frames with `read_frame` (the capture stream under test),
    then restores the original value. Returns PASS / FAIL / UNSUPPORTED.

    `read_frame` may return a frame, or (frame, capture_timestamp_ms) with the driver's
    CLOCK_MONOTONIC timestamp (cv2.CAP_PROP_POS_MSEC on V4L2). With timestamps, each frame is
    labelled with the command active at capture time, which removes the variable latency of
    the driver's frame queue (it grows at low frame rates).
    """
    cfg = {**CHALLENGE_CONFIG, **(cfg or {})}
    t0 = time.perf_counter()
    controls = query_controls(device_node)
    chosen = None
    for cname, cid in CHALLENGE_CONTROLS:
        ctl = next((v for v in controls.values() if v["id"] == cid), None)
        if ctl and ctl["type"] == V4L2_CTRL_TYPE_INTEGER and not ctl["flags"] & (
                V4L2_CTRL_FLAG_DISABLED | V4L2_CTRL_FLAG_READ_ONLY | V4L2_CTRL_FLAG_INACTIVE):
            chosen = (cname, cid, ctl)
            break
    if chosen is None:
        return {"verdict": "UNSUPPORTED", "passed": None, "control": None, "controls_available": sorted(controls),
                "details": "No writable brightness/gamma/gain control on this node; challenge skipped.",
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 1)}

    cname, cid, ctl = chosen
    try:
        fd = os.open(device_node, os.O_RDWR | os.O_NONBLOCK)
    except OSError as e:
        return {"verdict": "UNSUPPORTED", "passed": None, "control": cname,
                "details": f"Cannot open {device_node} for control access: {e.strerror}", "latency_ms": 0.0}

    attempts = []
    original = None
    try:
        original = _get_ctrl(fd, cid)
        step = max(1, ctl["step"])
        delta = max(step, int(round(cfg["delta_fraction"] * (ctl["max"] - ctl["min"]) / step)) * step)
        hi = min(ctl["max"], original + delta)
        lo = max(ctl["min"], original - delta)
        if hi - lo < 2 * step:
            return {"verdict": "UNSUPPORTED", "passed": None, "control": cname,
                    "details": f"Control {cname} has no usable range around {original}.", "latency_ms": 0.0}
        total = cfg["attempts"] * cfg["slots"] * cfg["frames_per_slot"]
        k = 0
        for _ in range(cfg["attempts"]):
            seq = make_challenge_sequence(cfg["slots"], cfg["min_sign_changes"], rng)
            luma, command = [], []
            reference = None
            set_times: List[tuple] = []  # (monotonic ms when the command was applied, command)
            for s in seq:
                _set_ctrl(fd, cid, hi if s > 0 else lo)
                set_times.append((time.monotonic() * 1000.0, s))
                for _ in range(cfg["frames_per_slot"]):
                    got = read_frame()
                    frame, ts = got if isinstance(got, tuple) else (got, None)
                    if frame is None:
                        break
                    k += 1
                    if on_frame:
                        on_frame(frame, k, total)
                    if ts is not None:
                        # Label the frame with the command in effect when the sensor captured it,
                        # so frames that waited in the driver queue are not mislabelled.
                        active = [c for (t_set, c) in set_times if t_set <= ts]
                        if not active:
                            continue  # captured before the challenge started
                        cmd = active[-1]
                    else:
                        cmd = s
                    if reference is None:
                        reference = _small_gray(frame)
                    luma.append(brightness_shift(frame, reference))
                    command.append(cmd)
            _set_ctrl(fd, cid, original)
            res = analyze_challenge_response(luma, command, cfg["max_lag"], cfg["min_corr"], cfg["min_effect"])
            res.update({"sequence": seq, "levels": [lo, hi], "frames": len(luma),
                        "luma": [round(v, 2) for v in luma], "command": command})
            attempts.append(res)
            if res["passed"]:
                break
            for _ in range(3):  # let the sensor settle before a retry
                read_frame()
    except OSError as e:
        return {"verdict": "UNSUPPORTED", "passed": None, "control": cname,
                "details": f"VIDIOC_S_CTRL on {cname} failed: {e.strerror}", "latency_ms": 0.0}
    finally:
        if original is not None:
            try:
                _set_ctrl(fd, cid, original)
            except OSError:
                pass
        os.close(fd)

    passed = any(a["passed"] for a in attempts)
    last = next((a for a in attempts if a["passed"]), attempts[-1])
    return {
        "verdict": "PASS" if passed else "FAIL",
        "passed": passed,
        "control": cname,
        "original_value": original,
        "levels": last["levels"],
        "corr": last["corr"],
        "effect": last["effect"],
        "lag_frames": last["lag"],
        "attempts": attempts,
        "details": (f"Frames tracked a random {cname} challenge on {device_node} "
                    f"(r={last['corr']:.2f}, effect={last['effect']:.1f} grey levels, lag={last['lag']} frames)."
                    if passed else
                    f"Frames did NOT respond to a {cname} challenge applied to {device_node} "
                    f"(best r={last['corr']:.2f}, effect={last['effect']:.1f}): the stream is not produced by this sensor."),
        "latency_ms": round((time.perf_counter() - t0) * 1000.0, 1),
    }


def capture_attested_frames(device_node: str, n_frames: int, width: int = 640, height: int = 480,
                            fourcc: str = "MJPG", warmup_frames: int = 15, challenge: bool = True,
                            on_frame: Optional[Callable[[np.ndarray, int, int, str], None]] = None) -> Dict[str, Any]:
    """
    Opens `device_node` (V4L2 backend, by path so the attested node is the captured node),
    lets auto-exposure settle, runs the active sensor challenge on the same stream, then
    captures `n_frames` raw frames for Layers 2 and 3. Frames are never re-encoded.
    """
    import cv2

    out: Dict[str, Any] = {"frames": [], "challenge": None, "error": None, "node": device_node}
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
                device_node, read_ts,
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
