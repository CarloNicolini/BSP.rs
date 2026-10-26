"""Paired CPU/Metal measurements using the installed native Python extension.

Build first: .venv/bin/maturin develop --release --features extension-module,metal
Run outside instrumentation/Metal validation. Profiling is a separate command.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import tempfile
import time

import numpy as np
from pybsp_rust import ExecutionOptions, Formula, Options, Solver, solve_many
from pybsp_rust import bsp_rust as core


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--max-n", type=int, default=1_000_000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--sweeps", type=int, default=30)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--full-seeds", type=int, default=10)
    parser.add_argument("--full-only", action="store_true")
    parser.add_argument("--full-large", action="store_true")
    parser.add_argument("--backends", nargs="+", default=["cpu", "cpu-parallel", "metal"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.repeats, args.sweeps, args.max_n) < 1 or args.full_seeds < 0:
        parser.error("positive limits and nonnegative full-seeds are required")
    with tempfile.TemporaryDirectory(prefix="bsp-build-info-") as directory:
        path=Path(directory)/"build.json"
        Solver(Formula([],n=0),execution=ExecutionOptions(profile=path)).solve()
        profile=json.loads(path.read_text())
        build={k:profile[k] for k in ["compiler","build_profile","rustflags","arch","os","package_version"]}
    binary=Path(core.__file__)
    rows = []
    cases = [(3, 4.2, n, "dense") for n in [400, 1000, 10000, 100000, 1000000] if n <= args.max_n]
    cases += [(4, 9.5, min(10000,args.max_n), "dense"), (6, 40, min(2000,args.max_n), "dense"), (12, 20, min(1000,args.max_n), "dense")]
    cases += [(3, 4.2, min(1000,args.max_n), "residual"), (3, 4.2, min(10000,args.max_n), "mixed")]
    if args.full_only:
        cases = []
    for k, alpha, n, kind in cases:
        formula = Formula.random(k, alpha, n, seed=7)
        if kind == "residual":
            clauses = formula.clauses
            for clause in clauses:
                if all(literal < 0 for literal in clause):
                    clause[0] *= -1
            formula = Formula(clauses,n=n)
        if kind == "mixed":
            formula = Formula(formula.clauses + [list(range(1,min(n,130)+1))], n=n)
        solvers = {b: Solver(formula, Options(damping=.2), execution=ExecutionOptions(backend=b,threads=args.threads)) for b in args.backends}
        indices = np.arange(formula.edges, dtype=np.uint64)
        messages = .05 + .25*((indices*37) % 997)/997
        del indices
        if kind == "residual":
            for s in solvers.values():
                for v in range(n//2):
                    s.fix(v, True)
        samples = {b: [] for b in solvers}
        errors = {}
        for trial in range(args.repeats+1):
            snapshots = {}
            order = list(solvers) if trial%2 else list(reversed(solvers))
            for backend in order:
                s = solvers[backend]
                s.set_messages(messages)
                start = time.perf_counter()
                try:
                    residual = s.sweep_many(args.sweeps)
                except RuntimeError as error:
                    errors[backend] = str(error)
                    continue
                elapsed = time.perf_counter()-start
                snapshots[backend] = s.messages
                assert np.isfinite(snapshots[backend]).all()
                if trial:
                    samples[backend].append(elapsed)
            if "cpu" in snapshots and "cpu-parallel" in snapshots:
                np.testing.assert_array_equal(snapshots["cpu"],snapshots["cpu-parallel"])
        differences = {b: float(np.max(np.abs(snapshots[b]-snapshots["cpu"]),initial=0)) for b in snapshots if "cpu" in snapshots}
        row = dict(k=k, alpha=alpha, n=n, edges=formula.edges, kind=kind, sweeps=args.sweeps,
                   samples=samples, seconds={b: statistics.median(v) for b,v in samples.items() if v},
                   max_message_error=differences, errors=errors,
                   execution={b:s.execution_report for b,s in solvers.items()})
        rows.append(row)
        print(json.dumps(row),flush=True)
        del solvers, formula, messages, snapshots
    full = []
    full_cases = [(3,4.15,400),(4,9.5,400)] if args.full_seeds else []
    if args.full_large and args.full_seeds:
        full_cases.append((3,4.15,10000))
    for k,alpha,n in full_cases:
        formulas = [Formula.random(k,alpha,n,seed=seed) for seed in range(args.full_seeds)]
        options = [Options(seed=seed,max_flips=100000,timeout_seconds=10) for seed in range(args.full_seeds)]
        for backend in args.backends:
            start = time.perf_counter()
            results = solve_many(formulas,options,execution=ExecutionOptions(backend=backend,threads=args.threads))
            elapsed = time.perf_counter()-start
            for f,r in zip(formulas,results,strict=True):
                if r.status == "sat": assert f.is_satisfied(r.assignment)
            full.append(dict(k=k,alpha=alpha,n=n,backend=backend,seconds=elapsed,
                             jobs=[dict(seed=r.seed,status=r.status,seconds=r.wall_seconds,sp_sweeps=r.sp_sweeps,flips=r.flips,execution=r.execution) for r in results]))
            print(json.dumps(full[-1]),flush=True)
    sources = [Path("Cargo.lock"),Path("Cargo.toml"),Path("build.rs"),Path(__file__)]+list(Path("src").glob("*.rs"))+list(Path("crates/bsp-metal/src").glob("*"))
    report = dict(platform=platform.platform(),cpu_count=os.cpu_count(),rustc=subprocess.check_output(["rustc","--version"],text=True).strip(),
                  configuration=vars(args)|{"output":str(args.output)},build=build,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),max_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},rows=rows,full=full)
    args.output.write_text(json.dumps(report,indent=2)+"\n")


if __name__ == "__main__":
    main()
