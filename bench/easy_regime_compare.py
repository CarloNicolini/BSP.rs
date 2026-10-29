"""Compare complete Python solve calls on identical random 3-SAT formulas.

Run with the C++ project's Python, with BSP.rs/python on PYTHONPATH. Metal
needs GPU access. No solver sources are modified. Configuration and formula
construction are outside the timer; solver setup and result conversion are in it.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import threading
import time

import numpy as np
import pybsp
from pybsp import _native as cpp_core
import pybsp_rust as rust
from pybsp_rust import bsp_rust as rust_core


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def run_case(backend, rf, cf, literals, seed, timeout):
    """Time one public Python call, then independently check every assignment."""
    if backend == "cpp":
        options = pybsp.SolverOptions(seed=seed, log_level="error")
        token = pybsp.CancellationToken()
        timer = threading.Timer(timeout, token.cancel)
        timer.daemon = True
        timer.start()
        start = time.perf_counter_ns()
        try:
            result = pybsp.solve(cf, options, cancel=token)
        except pybsp.SolverError as exc:
            result = exc.result
        finally:
            elapsed = (time.perf_counter_ns() - start) / 1e9
            timer.cancel()
        assignments = result.solutions
        expected = np.arange(1, rf.n + 1)
        encoding_ok = (assignments.ndim == 2 and assignments.shape[1] == rf.n
                       and np.all(np.abs(assignments) == expected))
        truth = assignments > 0
        extra = dict(reason=result.reason, steps=len(result.steps),
                     sp_sweeps=int(result.steps.sp_sweeps.iloc[-1]) if len(result.steps) else 0)
        claimed_success = result.status == "solved"
    else:
        options = rust.Options(seed=seed, timeout_seconds=timeout)
        execution = rust.ExecutionOptions(backend=backend, threads=1)
        start = time.perf_counter_ns()
        result = rust.solve(rf, options, execution=execution)
        elapsed = (time.perf_counter_ns() - start) / 1e9
        assignments = (np.empty((0, rf.n), dtype=np.int8) if result.assignment is None
                       else result.assignment.reshape(1, -1))
        encoding_ok = (assignments.ndim == 2 and assignments.shape[1] == rf.n
                       and np.all((assignments == 0) | (assignments == 1)))
        truth = assignments.astype(bool)
        extra = dict(steps=len(result.steps), sp_sweeps=result.sp_sweeps,
                     flips=result.flips, execution=result.execution)
        claimed_success = result.status == "sat"
    # This NumPy clause check does not call either solver's satisfaction checker.
    unsatisfied = []
    if encoding_ok:
        indices = np.abs(literals) - 1
        signs = literals > 0
        for values in truth:
            unsatisfied.append(int(np.count_nonzero(~np.any(values[indices] == signs, axis=1))))
    verified = bool(len(assignments)) and bool(encoding_ok) and all(x == 0 for x in unsatisfied)
    if claimed_success and not verified:
        raise AssertionError(f"{backend} returned an invalid successful result")
    row = dict(backend=backend, solver_seed=seed, seconds=elapsed,
               native_seconds=result.wall_seconds, status=result.status,
               solutions=len(assignments), encoding_ok=bool(encoding_ok),
               unsatisfied_clauses=unsatisfied, verified=verified, **extra)
    return row, truth


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[500, 1000, 2000, 5000, 10000, 20000])
    parser.add_argument("--formula-seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--small-repeats", type=int, help="override repetitions for n <= 2000")
    parser.add_argument("--min-seconds", type=float, default=5)
    parser.add_argument("--max-repeats", type=int, default=50)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--backends", nargs="+", default=["cpu", "metal", "cpp"])
    parser.add_argument("--pilot", action="store_true", help="one un-warmed run per formula/backend")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "formulas").mkdir(exist_ok=True)
    (args.output / "witnesses").mkdir(exist_ok=True)
    metadata = dict(
        platform=platform.platform(), python=platform.python_version(),
        cpu=command("sysctl", "-n", "machdep.cpu.brand_string"),
        ram_bytes=int(command("sysctl", "-n", "hw.memsize")),
        logical_cpus=int(command("sysctl", "-n", "hw.ncpu")),
        rust_commit=command("git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"),
        cpp_commit=command("git", "-C", str(Path(pybsp.__file__).resolve().parents[2]), "rev-parse", "HEAD"),
        binaries={str(m.__file__): sha256(m.__file__) for m in [cpp_core, rust_core]},
        script_sha256=sha256(__file__), numpy=np.__version__,
        configuration=vars(args) | {"output": str(args.output)},
        rust_options=asdict(rust.Options(seed=1001, timeout_seconds=args.timeout)),
        cpp_options=asdict(pybsp.SolverOptions(seed=1001, log_level="error")),
        environment={k: os.environ.get(k) for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]},
        timing="perf_counter_ns: public solve call, including solver setup/result conversion, excluding imports/formula construction/validation/witness saving",
        cpp_work="stock public API: up to ten WalkSAT solutions plus verification and whitening",
    )
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    raw = args.output / "runs.jsonl"
    if raw.exists():
        raise FileExistsError(f"Refusing to overwrite {raw}")
    # Warm each backend once, outside the measured workload. Each measured run
    # still creates a fresh solver, so native/GPU solver setup remains included.
    if not args.pilot:
        warm = rust.Formula.random(3, 4, 500, seed=123)
        rows = warm.clauses
        for backend in args.backends:
            row, _ = run_case(backend, warm, pybsp.Formula(rows, n=500), np.asarray(rows), 1001, args.timeout)
            print(json.dumps(dict(warmup=True, **row)), flush=True)
    with raw.open("a", buffering=1) as log:
        for n in args.sizes:
            cases = []
            totals = {backend: 0.0 for backend in args.backends}
            for formula_seed in args.formula_seeds:
                rf = rust.Formula.random(3, 4, n, seed=formula_seed)
                path = args.output / "formulas" / f"n{n}_seed{formula_seed}.cnf"
                path.write_text(rf.to_dimacs())
                clauses = rf.clauses
                assert rf.m == 4 * n and all(len(c) == 3 for c in clauses)
                cf = pybsp.Formula(clauses, n=n)
                assert cf.clauses == tuple(map(tuple, clauses))
                cases.append((formula_seed, rf, cf, np.asarray(clauses, dtype=np.int32), sha256(path)))
            repeats = (args.small_repeats if n <= 2000 and args.small_repeats is not None else args.repeats)
            if args.pilot:
                repeats = 1
            trial = 0
            while trial < repeats or (not args.pilot and min(totals.values()) < args.min_seconds and trial < args.max_repeats):
                for case_index, (formula_seed, rf, cf, literals, formula_hash) in enumerate(cases):
                    order = args.backends.copy()
                    random.Random(10000 * n + 100 * trial + case_index).shuffle(order)
                    for backend in order:
                        row, truth = run_case(backend, rf, cf, literals, 1000 + formula_seed, args.timeout)
                        witness = f"n{n}_f{formula_seed}_r{trial}_{backend}.npz"
                        np.savez_compressed(args.output / "witnesses" / witness, truth=truth)
                        row.update(n=n, m=4 * n, k=3, alpha=4.0, formula_seed=formula_seed,
                                   formula_sha256=formula_hash, repeat=trial, order=order,
                                   witness=f"witnesses/{witness}")
                        log.write(json.dumps(row) + "\n")
                        print(json.dumps({k: row[k] for k in ["n", "formula_seed", "repeat", "backend", "status", "verified", "solutions", "seconds", "sp_sweeps"]}), flush=True)
                        totals[backend] += row["seconds"]
                trial += 1


if __name__ == "__main__":
    main()
