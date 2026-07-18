"""Lightweight system-resource sampler, stamped onto OTel spans.

Samples CPU + RAM (psutil), NIC counters (psutil, cumulative -> per-second rates),
and GPU (pynvml on desktop NVIDIA; tegrastats on Jetson; omitted otherwise), then
stamps them as span attributes. The ClickHouse trace store then carries per-host
CPU/GPU/RAM/network without any separate metrics pipeline — `metrics_harvest.py`
reads them back with its `resource_usage` block.

Host/tier labels come from AHPECA_HOST_LABEL / AHPECA_TIER (ssh-alias-style strings,
never IPs). Cheap by design: the whole sample is TTL-cached (~1s), so stamping on a
frame-rate span costs one dict copy. Nothing here ever raises into the caller —
telemetry must never break the app path.

NOTE: this file is duplicated verbatim across the backend, supervision, and drone
code roots (they share no package). Keep the copies in sync.
"""
import os
import socket
import subprocess
import threading
import time

try:
    import psutil
except Exception:  # psutil optional; resource attrs simply won't be stamped
    psutil = None

_HOST = os.getenv("AHPECA_HOST_LABEL") or socket.gethostname()
_TIER = os.getenv("AHPECA_TIER", "")
_TTL_S = float(os.getenv("AHPECA_RESOURCE_TTL_S", "1.0"))

_lock = threading.Lock()
_cache = {"t": 0.0, "vals": {}}
_net_prev = None            # (t, tx, rx, ptx, prx, err, drop)
_gpu = None                 # None=undetected, then "nvml" | "tegra" | "none"
_nvml_handle = None
_tegra_vals = {}


# ---- network (cumulative counters -> per-second rates) ---------------------

def _net_rates(now):
    global _net_prev
    if psutil is None:
        return {}
    try:
        io = psutil.net_io_counters()
    except Exception:
        return {}
    cur = (now, io.bytes_sent, io.bytes_recv, io.packets_sent, io.packets_recv,
           io.errin + io.errout, io.dropin + io.dropout)
    prev, _net_prev = _net_prev, cur
    if prev is None:
        return {}
    dt = max(cur[0] - prev[0], 1e-6)
    return {
        "net.bytes_sent_per_s": round((cur[1] - prev[1]) / dt, 1),
        "net.bytes_recv_per_s": round((cur[2] - prev[2]) / dt, 1),
        "net.packets_sent_per_s": round((cur[3] - prev[3]) / dt, 1),
        "net.packets_recv_per_s": round((cur[4] - prev[4]) / dt, 1),
        "net.err_per_s": round((cur[5] - prev[5]) / dt, 2),
        "net.drop_per_s": round((cur[6] - prev[6]) / dt, 2),
    }


# ---- GPU (auto-detect pynvml, else tegrastats, else none) ------------------

def _parse_tegra(line):
    """Pull GPU util (GR3D_FREQ) and RAM use out of one tegrastats line."""
    import re
    out = {}
    m = re.search(r"GR3D_FREQ (\d+)%", line)
    if m:
        out["gpu.util.percent"] = float(m.group(1))
    m = re.search(r"RAM (\d+)/(\d+)MB", line)
    if m:
        used, total = float(m.group(1)), float(m.group(2))
        out["gpu.mem.used_mb"] = used            # Tegra shares system RAM with the GPU
        out["gpu.mem.total_mb"] = total
        out["gpu.mem.percent"] = round(100 * used / total, 1) if total else 0.0
    return out


def _start_tegra():
    def loop():
        global _tegra_vals
        try:
            p = subprocess.Popen(["tegrastats", "--interval", "1000"],
                                 stdout=subprocess.PIPE, text=True)
        except Exception:
            return
        for line in p.stdout:
            try:
                _tegra_vals = _parse_tegra(line)
            except Exception:
                pass
    threading.Thread(target=loop, daemon=True).start()


def _init_gpu():
    global _gpu, _nvml_handle
    if _gpu is not None:
        return
    try:
        import pynvml
        pynvml.nvmlInit()
        _nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        _gpu = "nvml"
        return
    except Exception:
        pass
    try:
        if subprocess.run(["which", "tegrastats"], capture_output=True).returncode == 0:
            _gpu = "tegra"
            _start_tegra()
            return
    except Exception:
        pass
    _gpu = "none"


def _gpu_sample():
    _init_gpu()
    if _gpu == "nvml":
        try:
            import pynvml
            u = pynvml.nvmlDeviceGetUtilizationRates(_nvml_handle)
            m = pynvml.nvmlDeviceGetMemoryInfo(_nvml_handle)
            return {
                "gpu.util.percent": float(u.gpu),
                "gpu.mem.used_mb": round(m.used / 1e6, 1),
                "gpu.mem.total_mb": round(m.total / 1e6, 1),
                "gpu.mem.percent": round(100 * m.used / m.total, 1) if m.total else 0.0,
            }
        except Exception:
            return {}
    if _gpu == "tegra":
        return dict(_tegra_vals)
    return {}


# ---- public API ------------------------------------------------------------

def sample():
    """Return a TTL-cached dict of resource attributes (keys omitted if unavailable)."""
    now = time.time()
    with _lock:
        if now - _cache["t"] < _TTL_S and _cache["vals"]:
            return dict(_cache["vals"])
        vals = {}
        if psutil is not None:
            try:
                vals["cpu.percent"] = round(psutil.cpu_percent(interval=None), 1)
                vm = psutil.virtual_memory()
                vals["mem.used_mb"] = round(vm.used / 1e6, 1)
                vals["mem.percent"] = round(vm.percent, 1)
            except Exception:
                pass
            vals.update(_net_rates(now))
        vals.update(_gpu_sample())
        _cache["t"] = now
        _cache["vals"] = vals
        return dict(vals)


def stamp(span):
    """Stamp resource.host/tier + the current sample onto an existing span. Never raises."""
    try:
        if span is None:
            return
        span.set_attribute("resource.host", _HOST)
        if _TIER:
            span.set_attribute("resource.tier", _TIER)
        for k, v in sample().items():
            span.set_attribute(k, v)
    except Exception:
        pass


def start_periodic_emitter(tracer, interval=None, service=None):
    """Emit a `resource.sample` span every `interval` seconds. For hosts with no app
    span to stamp onto (e.g. the obs plane). Returns the daemon thread."""
    period = interval or float(os.getenv("AHPECA_RESOURCE_INTERVAL_S", "5"))

    def loop():
        while True:
            try:
                with tracer.start_as_current_span("resource.sample") as sp:
                    if service:
                        sp.set_attribute("resource.service", service)
                    stamp(sp)
            except Exception:
                pass
            time.sleep(period)

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t


# Prime psutil's since-last-call counters so the first real sample is meaningful.
if psutil is not None:
    try:
        psutil.cpu_percent(interval=None)
        _net_rates(time.time())
    except Exception:
        pass
