"""Audit saved CNFs and solution witnesses, then summarize paired solve timings."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    formula_cache = {}
    for directory in args.directories:
        for line in (directory / "runs.jsonl").read_text().splitlines():
            row = json.loads(line)
            path = directory / "formulas" / f"n{row['n']}_seed{row['formula_seed']}.cnf"
            if path not in formula_cache:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                lines = path.read_text().splitlines()
                assert lines[0] == f"p cnf {row['n']} {4 * row['n']}"
                clauses = np.asarray([list(map(int, s.split())) for s in lines[1:]], dtype=np.int32)
                assert clauses.shape == (row['m'], 4) and np.all(clauses[:, -1] == 0)
                formula_cache[path] = digest, clauses[:, :3]
            digest, literals = formula_cache[path]
            assert digest == row["formula_sha256"]
            with np.load(directory / row["witness"]) as archive:
                truth = archive["truth"]
            assert truth.dtype == np.bool_ and truth.shape == (row["solutions"], row["n"])
            bad = [int(np.count_nonzero(~np.any(v[np.abs(literals) - 1] == (literals > 0), axis=1))) for v in truth]
            assert bad == row["unsatisfied_clauses"]
            if len(truth):
                assert row["encoding_ok"] and not any(bad)
            if row["verified"]:
                assert row["encoding_ok"] and len(truth) and not any(bad)
                assert row["status"] in ["sat", "solved"]
            if row["backend"] == "metal":
                execution = row["execution"]
                assert execution["requested_backend"] == "metal"
                assert execution["backend"] == "metal" and not execution["fallbacks"]
                assert execution["device_error"] is None and execution["metal_sweeps"] > 0
            rows.append(row)
    sizes = sorted({r["n"] for r in rows})
    summary = []
    for n in sizes:
        group = [r for r in rows if r["n"] == n]
        seeds = sorted({r["formula_seed"] for r in group})
        per_backend = {}
        for backend in ["cpu", "metal", "cpp"]:
            records = [r for r in group if r["backend"] == backend]
            solved = [r for r in records if r["verified"]]
            per_formula = {}
            for seed in seeds:
                samples = [r["seconds"] for r in solved if r["formula_seed"] == seed]
                if samples:
                    per_formula[seed] = statistics.median(samples)
            per_backend[backend] = dict(
                calls=len(records), verified_calls=len(solved),
                statuses={s: sum(r["status"] == s for r in records) for s in {r["status"] for r in records}},
                validated_assignments=sum(r["solutions"] for r in solved),
                seconds_median=statistics.median(r["seconds"] for r in solved) if solved else None,
                seconds_iqr=np.percentile([r["seconds"] for r in solved], [25, 75]).tolist() if solved else None,
                per_formula_medians=per_formula,
                per_formula_repeat_relative_mad={seed: (statistics.median(abs(r["seconds"] - med) for r in solved if r["formula_seed"] == seed) / med if sum(r["formula_seed"] == seed for r in solved) > 1 else None) for seed, med in per_formula.items()},
            )
        common = sorted(set.intersection(*(set(b["per_formula_medians"]) for b in per_backend.values())))
        # Equal weight per formula; repeated timings reduce measurement noise.
        means = {backend: statistics.mean(data["per_formula_medians"][s] for s in common) if common else None for backend, data in per_backend.items()}
        summary.append(dict(n=n, m=4*n, formulas=len(seeds), common_verified_formulas=len(common),
                            common_seeds=common, paired_seconds=means, backends=per_backend,
                            metal_speedup_cpu=means["cpu"] / means["metal"] if common else None,
                            rust_cpu_speedup_cpp=means["cpp"] / means["cpu"] if common else None,
                            metal_speedup_cpp=means["cpp"] / means["metal"] if common else None))
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(verified_calls=sum(r["verified"] for r in rows), calls=len(rows),
                  validated_assignments=sum(r["solutions"] for r in rows if r["verified"]),
                  invalid_assignments=0, cpu_fallbacks=0, rows=summary)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    flat = []
    for r in summary:
        flat.append(dict(n=r["n"], m=r["m"], common_verified_formulas=r["common_verified_formulas"],
                         **{f"{b}_seconds": r["paired_seconds"][b] for b in ["cpu", "metal", "cpp"]},
                         **{f"{b}_verified_calls": r["backends"][b]["verified_calls"] for b in ["cpu", "metal", "cpp"]},
                         **{f"{b}_calls": r["backends"][b]["calls"] for b in ["cpu", "metal", "cpp"]},
                         metal_speedup_cpu=r["metal_speedup_cpu"], metal_speedup_cpp=r["metal_speedup_cpp"]))
    with (args.output / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(flat)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
