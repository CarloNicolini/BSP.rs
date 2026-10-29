"""Measure an isolated C++ build with one/ten outputs and optional whitening.

Reuse the exact CNFs, seeds, and small-case repetition counts of the initial
campaign. Existing Rust measurements remain the one-solution reference.
"""
import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import statistics
import sys
import threading
import time

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--numsol", type=int, choices=[1, 10], default=1)
    parser.add_argument("--whiten", type=int, choices=[0, 1], default=0)
    parser.add_argument("--sizes", type=int, nargs="+", default=[500, 1000, 2000, 5000, 10000, 20000])
    parser.add_argument("--formula-seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--repeats", type=int, help="override original campaign's repetition counts")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["BSP_BENCH_NUMSOL"] = str(args.numsol)
    os.environ["BSP_BENCH_WHITEN"] = str(args.whiten)
    # pybsp.__init__ is lazy; replace only the native module before importing
    # the public Formula/SolverOptions/solve wrappers from the unchanged package.
    import pybsp
    assert "pybsp._native" not in sys.modules
    spec = importlib.util.spec_from_file_location("pybsp._native", args.extension)
    native = importlib.util.module_from_spec(spec)
    sys.modules["pybsp._native"] = native
    pybsp._native = native
    spec.loader.exec_module(native)
    from pybsp import Formula, SolverOptions, CancellationToken, solve
    old_rows = [json.loads(x) for x in (args.original / "runs.jsonl").read_text().splitlines()]
    metadata = dict(configuration={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    extension_sha256=digest(args.extension), script_sha256=digest(Path(__file__)),
                    original_metadata=json.loads((args.original / "metadata.json").read_text()),
                    cpp_options=asdict(SolverOptions(seed=1001, log_level="error")),
                    note="Isolated source build: output target and whitening changed, phase clocks added. Same unchanged public Python wrapper. Rust references are the original one-solution measurements.")
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (args.output / "witnesses").mkdir()
    records = []

    def run(formula, literals, seed):
        options = SolverOptions(seed=1000 + seed, log_level="error")
        token = CancellationToken()
        timer = threading.Timer(args.timeout, token.cancel)
        timer.daemon = True
        timer.start()
        start = time.perf_counter_ns()
        try:
            result = solve(formula, options, cancel=token)
        finally:
            elapsed = (time.perf_counter_ns() - start) / 1e9
            timer.cancel()
        assignments = result.solutions
        assert assignments.shape == (args.numsol, formula.n), (result.status, result.reason, assignments.shape)
        assert np.all(np.abs(assignments) == np.arange(1, formula.n + 1))
        truth = assignments > 0
        bad = [int(np.count_nonzero(~np.any(v[np.abs(literals) - 1] == (literals > 0), axis=1))) for v in truth]
        assert result.status == "solved" and not any(bad)
        assert len(result.whitening) == args.numsol * args.whiten
        phases = {k: float(v) for k, v in re.findall(r"BSP_BENCH (\w+) ([^\s]+)", result.stderr)}
        assert set(phases) == {"bsp_seconds", "walksat_first_seconds", "walksat_all_seconds", "postprocess_seconds", "whitening_seconds"}, result.stderr
        assert args.whiten or phases["whitening_seconds"] == 0
        return dict(seconds=elapsed, native_seconds=result.wall_seconds, phases=phases,
                    status=result.status, solutions=len(assignments), verified=True,
                    unsatisfied_clauses=bad, steps=len(result.steps),
                    sp_sweeps=int(result.steps.sp_sweeps.iloc[-1])), truth

    warm_path = args.original / "formulas/n500_seed1.cnf"
    warm = Formula.from_file(warm_path)
    warm_row, _ = run(warm, np.asarray(warm.clauses), 1)
    print(json.dumps(dict(warmup=True, **warm_row)), flush=True)
    with (args.output / "runs.jsonl").open("a", buffering=1) as log:
        for n in args.sizes:
            for seed in args.formula_seeds:
                path = args.original / "formulas" / f"n{n}_seed{seed}.cnf"
                formula = Formula.from_file(path)
                literals = np.asarray(formula.clauses, dtype=np.int32)
                previous = [r for r in old_rows if r["n"] == n and r["formula_seed"] == seed and r["backend"] == "cpp"]
                expected_hash = previous[0]["formula_sha256"]
                assert digest(path) == expected_hash
                with np.load(args.original / previous[0]["witness"]) as archive:
                    old_truth = archive["truth"].copy()
                repeats = args.repeats if args.repeats is not None else len(previous)
                for trial in range(repeats):
                    row, truth = run(formula, literals, seed)
                    row.update(n=n, formula_seed=seed, solver_seed=1000 + seed, repeat=trial,
                               numsol=args.numsol, whiten=args.whiten, formula_sha256=expected_hash,
                               original_first_assignment_equal=bool(np.array_equal(truth[0], old_truth[0])),
                               original_all_assignments_equal=bool(np.array_equal(truth, old_truth)) if args.numsol == 10 else None,
                               original_sp_sweeps_equal=row["sp_sweeps"] == previous[0]["sp_sweeps"])
                    witness = f"witnesses/n{n}_f{seed}_r{trial}.npz"
                    np.savez_compressed(args.output / witness, truth=truth)
                    row["witness"] = witness
                    log.write(json.dumps(row) + "\n")
                    records.append(row)
                    print(json.dumps(row), flush=True)
    summary = []
    for n in args.sizes:
        group = [r for r in records if r["n"] == n]
        medians = [statistics.median(r["seconds"] for r in group if r["formula_seed"] == seed) for seed in args.formula_seeds]
        summary.append(dict(n=n, cpp_seconds=statistics.mean(medians),
                            formulas=len(args.formula_seeds), verified_calls=len(group)))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
