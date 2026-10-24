"""Record complete runs on shared inputs; do not infer a kernel speedup.

C++ collects ten solutions and whitens them; Rust returns one checked solution.
Their search trajectories differ. Each worker has its own native process state.
"""
import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import threading
import time


def worker(backend, path, seed, timeout):
    if backend == "rust":
        from pybsp_rust import Formula, Options, solve
        f = Formula.from_dimacs(Path(path).read_text())
        started = time.perf_counter()
        r = solve(f, Options(seed=seed, timeout_seconds=timeout))
        elapsed = time.perf_counter()-started
        return {"status":r.status,"seconds":elapsed,"sp_sweeps":r.sp_sweeps,
            "steps":len(r.steps),"solutions":int(r.assignment is not None),"decimations":r.decimations,
            "releases":r.releases,"verified":r.assignment is not None and f.is_satisfied(r.assignment)}
    from pybsp import Formula, SolverOptions, CancellationToken, solve
    f = Formula.from_file(path)
    token = CancellationToken()
    timer = threading.Timer(timeout, token.cancel)
    started = time.perf_counter()
    timer.start()
    try:
        r = solve(f, SolverOptions(seed=seed,log_level="error"), cancel=token)
    finally:
        timer.cancel()
    return {"status":r.status,"reason":r.reason,"seconds":time.perf_counter()-started,
        "sp_sweeps_last_fixed_point":int(r.steps.sp_sweeps.iloc[-1]) if len(r.steps) else None,
        "steps":len(r.steps),"solutions":len(r.solutions),
        "verified":bool(len(r.solutions)) and all(f.is_satisfied(a) for a in r.solutions)}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--worker", choices=["cpp","rust"])
    parser.add_argument("--formula")
    parser.add_argument("--seed", type=int,default=3)
    parser.add_argument("--timeout",type=float,default=5)
    parser.add_argument("--output",type=Path)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.worker,args.formula,args.seed,args.timeout)))
        return
    from pybsp_rust import Formula
    rows=[]
    with tempfile.TemporaryDirectory(prefix="bsp-solve-compare-") as directory:
        for k,alpha,n,seed in [(3,2.5,400,3),(3,4.15,400,3),(3,4.15,400,2),(4,9.5,400,3)]:
            path=Path(directory)/"formula.cnf"
            path.write_text(Formula.random(k,alpha,n,seed).to_dimacs())
            row={"k":k,"alpha":alpha,"n":n,"seed":seed}
            for backend in ["cpp","rust"]:
                command=[sys.executable,__file__,"--worker",backend,"--formula",str(path),"--seed",str(seed),"--timeout",str(args.timeout)]
                try:
                    row[backend]=json.loads(subprocess.check_output(command,text=True,timeout=args.timeout+15))
                except subprocess.TimeoutExpired:
                    row[backend]={"status":"worker_timeout"}
            rows.append(row)
    report={"platform":platform.platform(),"note":__doc__.strip(),"rows":rows}
    text=json.dumps(report,indent=2)+"\n"
    print(text,end="")
    if args.output: args.output.write_text(text)


if __name__ == "__main__":
    main()
