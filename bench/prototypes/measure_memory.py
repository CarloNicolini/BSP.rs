"""Measure child RSS and macOS lifetime peak physical footprint.

Usage: python3 bench/prototypes/measure_memory.py BINARY OUTPUT.json [N]
An explicit N measures only the scalar backend at that size.
RSS units are bytes on macOS and KiB on Linux. This exercises one SP sweep,
not convergence, graph mutation, WalkSAT, result serialization, or GPU memory.
On macOS, sample proc_pid_rusage every 50 ms; report the largest lifetime peak
observed before exit. wait4 RSS alone excludes compressed/swapped pages.
"""

import json
import ctypes
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time


# Layout from sys/resource.h, rusage_info_v4: 16 UUID bytes + 35 u64s.
# proc_pid_rusage and footprint accounting are described in Apple's WWDC22
# "Profile and optimize your game's memory". Query only our own child.
class DarwinUsage(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16), ("values", ctypes.c_uint64 * 35)]


libproc = None
if platform.system() == "Darwin":
    libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    libproc.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]


def vm_state():
    return (subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
            if platform.system() == "Darwin" else None)


binary = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2])
rows = []
cases = ([(int(sys.argv[3]), "cpu")] if len(sys.argv) > 3 else
         [(100_000, "cpu"), (1_000_000, "cpu"), (1_000_000, "cpu-parallel")])
for n, backend in cases:
    command = [str(binary), "--backend", backend, "--threads", "4",
               "--generate", "3", "4.2", str(n), "--sweeps", "1"]
    with tempfile.TemporaryFile(mode="w+") as stdout, tempfile.TemporaryFile(mode="w+") as stderr:
        vm_before = vm_state()
        sampled_rss = sampled_footprint = lifetime_footprint = samples = 0
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr)
        while True:
            if libproc is not None:
                info = DarwinUsage()
                if libproc.proc_pid_rusage(process.pid, 4, ctypes.byref(info)) == 0:
                    sampled_rss = max(sampled_rss, info.values[6])
                    sampled_footprint = max(sampled_footprint, info.values[7])
                    lifetime_footprint = max(lifetime_footprint, info.values[28])
                    samples += 1
            child, status, usage = os.wait4(process.pid, os.WNOHANG)
            if child:
                break
            time.sleep(0.05)
        process.returncode = os.waitstatus_to_exitcode(status)
        stdout.seek(0)
        stderr.seek(0)
        if process.returncode:
            raise RuntimeError(stderr.read())
        result = json.load(stdout)
        vm_after = vm_state()
    rows.append({
        "command": command,
        "maximum_rss_bytes": usage.ru_maxrss * (1 if platform.system() == "Darwin" else 1024),
        "sampled_maximum_resident_bytes": sampled_rss if samples else None,
        "sampled_maximum_phys_footprint_bytes": sampled_footprint if samples else None,
        "observed_lifetime_peak_phys_footprint_bytes": lifetime_footprint if samples else None,
        "footprint_samples": samples,
        "vm_stat_before": vm_before,
        "vm_stat_after": vm_after,
        "solver_report": result,
    })
output.write_text(json.dumps({"platform": platform.platform(), "rows": rows}, indent=2) + "\n")
