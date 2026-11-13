"""Calibrate native BSP on disjoint seeds and verify generated SAT witnesses."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess

from net_analysis import verify_witness


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    binary = args.binary.resolve()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    jobs = [(n, a, seed) for n in (640, 1280) for a in (9.25, 9.45, 9.65, 9.8) for seed in (201, 202)]

    def run(job):
        n, alpha, seed = job
        name = f"{n}-{alpha}-{seed}"
        cnf = output / f"{name}.cnf"
        record = output / f"{name}.json"
        subprocess.run(
            [binary, "--generate", "4", str(alpha), str(n), "--seed", str(seed), "--write-formula", cnf],
            capture_output=True, check=True,
        )
        command = [
            binary, cnf, "--seed", "23", "--r", "0.7", "--frac", str(1 / (n + 1)),
            "--eps", "1e-5", "--damping", "0.2", "--max-sweeps", "4096", "--max-steps", "12000",
            "--max-flips", "4000000", "--no-progress", "--output", record,
        ]
        outcome = subprocess.run(command, capture_output=True, text=True)
        if outcome.returncode not in (0, 1) or not record.exists():
            raise RuntimeError(outcome.stderr)
        report = json.loads(record.read_text())
        verify_witness(cnf, n, {"status": report["status"], "witness": report["assignment"]})
        return {
            "n": n, "alpha": alpha, "seed": seed, "status": report["status"],
            "sweeps": report["sp_sweeps"], "steps": len(report["steps"]),
            "wall_seconds": report["wall_seconds"], "flips": report["flips"],
        }

    with ThreadPoolExecutor(args.workers) as pool:
        print(json.dumps(list(pool.map(run, jobs)), indent=2))


if __name__ == "__main__":
    main()
