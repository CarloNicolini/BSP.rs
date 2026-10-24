"""Matched SP sweeps; subprocess launch and graph construction are outside timers.

Both kernels start from the same DIMACS and per-edge messages. C++ uses its
existing graph storage and the extracted scalar loop, without convergence or
diagnostics overhead. Complete solves are a separate measurement.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import tempfile


def output(command):
    return json.loads(subprocess.check_output(list(map(str, command)), text=True))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--rust", type=Path, default=Path("target/release/bsp-rust"))
    parser.add_argument("--cpp", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--sweeps", type=int, default=30)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--build-note", default="Build flags unspecified; see PERFORMANCE.md for the reference build.")
    args = parser.parse_args()
    if args.repeats < 1 or args.sweeps < 1:
        parser.error("repeats and sweeps must be positive")
    rows = []
    for k, alpha, n in [(3,4.2,10_000), (4,9.5,10_000), (6,40,2_000), (12,20,1_000)]:
        with tempfile.TemporaryDirectory(prefix="bsp-rust-bench-") as directory:
            cnf = Path(directory)/"formula.cnf"
            subprocess.run([str(args.rust), "--seed", "7", "--generate", str(k), str(alpha), str(n), "--write-formula", str(cnf)], check=True)
            edges = int(alpha*n)*k
            messages = Path(directory)/"messages.json"
            messages.write_text(json.dumps([.05+.25*((e*37)%997)/997 for e in range(edges)]))
            commands = {
                "rust": [args.rust, "--sweeps", args.sweeps, "--damping", .2, "--messages", messages, "--dump-messages", cnf],
                "cpp": [args.cpp, cnf, args.sweeps, .2],
            }
            times = {key: [] for key in commands}
            answers = {}
            for repeat in range(args.repeats+1):
                # Alternate order, and discard the first round as warmup.
                for key in (["cpp","rust"] if repeat%2 else ["rust","cpp"]):
                    result = output(commands[key]); answers[key] = result
                    if repeat: times[key].append(result["seconds"])
            error = max(abs(a-b) for a,b in zip(answers["rust"]["messages"],answers["cpp"]["messages"],strict=True))
            finite = all(math.isfinite(v) for answer in answers.values() for v in answer["messages"])
            if not finite or error > 1e-9:
                raise RuntimeError(f"message mismatch for k={k}: {error}")
            medians = {key:statistics.median(value) for key,value in times.items()}
            rows.append({"k":k,"alpha":alpha,"n":n,"edges":edges,"sweeps":args.sweeps,"seconds":medians,
                "speedup":medians["cpp"]/medians["rust"],"max_message_error":error,"samples":times})
    report = {"machine":platform.machine(),"platform":platform.platform(),"cpu_count":os.cpu_count(),
        "rustc":subprocess.check_output(["rustc","--version"],text=True).strip(),
        "cpp_compiler":subprocess.check_output(["c++","--version"],text=True).splitlines()[0],
        "build_note":args.build_note,"damping":.2,"initial_messages":"0.05 + 0.25*((edge*37)%997)/997",
        "sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [args.rust,args.cpp,Path("src/solver.rs"),Path("bench/cpp_kernel.cpp"),Path("Cargo.lock")]},
        "rows":rows}
    text = json.dumps(report,indent=2)+"\n"
    print(text,end="")
    if args.output: args.output.write_text(text)


if __name__ == "__main__":
    main()
