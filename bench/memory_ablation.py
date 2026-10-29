"""Sequential CPU/Metal ablations with macOS physical-footprint accounting.

Each plan item specifies binary, formula, backend, mode, count, timeout, seed,
and output (a prefix). Timings come from the native harness; process memory
includes parsing/setup. Metal buffers occupy unified memory, not separate VRAM.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import time


class DarwinUsage(ctypes.Structure):
    # sys/resource.h: struct rusage_info_v4.
    _fields_ = [("uuid", ctypes.c_uint8 * 16), ("values", ctypes.c_uint64 * 35)]


def vm_state():
    return subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout


def verify_cnf(path, witness):
    values = Path(witness).read_bytes()
    if any(v > 1 for v in values):
        return False
    clauses = 0
    satisfied = False
    pending = False
    with open(path) as file:
        for line in file:
            if not line.strip() or line.startswith("c"):
                continue
            if line.startswith("p"):
                _, _, n, expected = line.split()
                if len(values) != int(n):
                    return False
                continue
            for token in line.split():
                lit = int(token)
                if lit == 0:
                    if not satisfied:
                        return False
                    clauses += 1
                    satisfied = pending = False
                else:
                    pending = True
                    satisfied |= values[abs(lit) - 1] == int(lit > 0)
    return not pending and clauses == int(expected)


def run(job):
    prefix = Path(job["output"])
    prefix.parent.mkdir(parents=True, exist_ok=True)
    destination = prefix.with_suffix(".json")
    if destination.exists():
        print(f"skip existing {destination}", flush=True)
        return
    command = [job["binary"], job["formula"], job["backend"], job["mode"],
               str(job.get("count", 0)), str(job["timeout"]), str(job["seed"]), str(prefix)]
    lib = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    lib.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    before = vm_state()
    peak_rss = peak_footprint = lifetime_footprint = samples = 0
    started = time.monotonic()
    print(f"start {prefix.name} {job['backend']} {job['mode']}", flush=True)
    with prefix.with_suffix(".stdout").open("w+") as stdout, prefix.with_suffix(".stderr").open("w+") as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr)
        killed = False
        while True:
            info = DarwinUsage()
            if lib.proc_pid_rusage(process.pid, 4, ctypes.byref(info)) == 0:
                peak_rss = max(peak_rss, info.values[6])
                peak_footprint = max(peak_footprint, info.values[7])
                lifetime_footprint = max(lifetime_footprint, info.values[28])
                samples += 1
            child, status, usage = os.wait4(process.pid, os.WNOHANG)
            if child:
                break
            if not killed and time.monotonic() - started > job["timeout"] + 60:
                os.kill(process.pid, signal.SIGKILL)
                killed = True
            time.sleep(0.05)
        process.returncode = os.waitstatus_to_exitcode(status)
        stdout.seek(0)
        stderr.seek(0)
        text = stdout.read()
        native = json.loads(text) if process.returncode == 0 else None
        errors = stderr.read()
    verify_seconds = None
    independent_verified = None
    if native is not None and native["report"].get("status") == "sat":
        check_started = time.monotonic()
        independent_verified = verify_cnf(job["formula"], prefix.with_suffix(".assignment.bin"))
        verify_seconds = time.monotonic() - check_started
        if not independent_verified:
            raise RuntimeError(f"independent witness check failed: {prefix}")
    result = dict(job=job, command=command, platform=platform.platform(),
        process_seconds=time.monotonic() - started, returncode=process.returncode,
        hard_timeout=killed, error=errors, native=native,
        maximum_rss_bytes=usage.ru_maxrss, sampled_peak_resident_bytes=peak_rss,
        sampled_peak_phys_footprint_bytes=peak_footprint,
        observed_lifetime_peak_phys_footprint_bytes=lifetime_footprint,
        footprint_samples=samples, vm_stat_before=before, vm_stat_after=vm_state(),
        independent_verified=independent_verified, independent_verify_seconds=verify_seconds,
        binary_sha256=hashlib.sha256(Path(job["binary"]).read_bytes()).hexdigest())
    destination.write_text(json.dumps(result, indent=2) + "\n")
    status = native["report"]["status"] if native else f"exit_{process.returncode}"
    print(f"done {prefix.name} status={status} footprint={lifetime_footprint/1e6:.1f}MB", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    for job in json.loads(args.plan.read_text()):
        run(job)


if __name__ == "__main__":
    main()
