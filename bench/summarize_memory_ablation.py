"""Check matched work and summarize native timing and process footprint."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    comparisons = []
    errors = []
    groups = defaultdict(list)
    for kind in ("actions", "kernels", "large-actions", "k4-actions"):
        results = {}
        for path in (args.root / kind).glob("*.json"):
            if path.name.endswith(".steps.json"):
                continue
            result = json.loads(path.read_text())
            variant = Path(result["job"]["binary"]).name
            job = result["job"]
            key = path.stem[len(variant) + 1:]
            results[variant, key] = (path, result)
        for (variant, key), (path, row) in results.items():
            if variant == "baseline" or ("baseline", key) not in results:
                continue
            base_path, base = results["baseline", key]
            if not row["native"] or not base["native"]:
                errors.append(str(path) + ": process failed")
                continue
            a, b = base["native"]["report"], row["native"]["report"]
            matched = (a["status"] == b["status"] and
                       a["status"] in ("ok", "paramagnetic", "sat") and
                       a["message_hash"] == b["message_hash"])
            if a["status"] == "sat":
                matched &= (base["independent_verified"] is True and
                            row["independent_verified"] is True and
                            digest(base_path.with_suffix(".assignment.bin")) ==
                            digest(path.with_suffix(".assignment.bin")))
            if kind != "kernels":
                matched &= json.loads(base_path.with_suffix(".steps.json").read_text()) == json.loads(path.with_suffix(".steps.json").read_text())
            else:
                matched &= a["residual"] == b["residual"]
            if not matched:
                errors.append(str(path) + ": trajectory or message mismatch")
            comparison = dict(kind=kind, variant=variant, key=key,
                backend=row["job"]["backend"], matched=bool(matched),
                speedup=a["compute_seconds"] / b["compute_seconds"],
                seconds=b["compute_seconds"], baseline_seconds=a["compute_seconds"],
                footprint_bytes=row["observed_lifetime_peak_phys_footprint_bytes"],
                baseline_footprint_bytes=base["observed_lifetime_peak_phys_footprint_bytes"],
                backend_extra_bytes=b["execution"]["allocated_bytes"],
                gpu_buffer_bytes=b["execution"]["allocated_bytes"] if row["job"]["backend"] == "metal" else 0,
                baseline_gpu_buffer_bytes=a["execution"]["allocated_bytes"] if row["job"]["backend"] == "metal" else 0)
            comparisons.append(comparison)
            groups[kind, row["job"]["backend"], variant].append(comparison)
    summary = []
    for (kind, backend, variant), rows in sorted(groups.items()):
        summary.append(dict(kind=kind, backend=backend, variant=variant, samples=len(rows),
            median_paired_speedup=statistics.median(r["speedup"] for r in rows),
            median_footprint_mb=statistics.median(r["footprint_bytes"] for r in rows)/1e6,
            median_baseline_footprint_mb=statistics.median(r["baseline_footprint_bytes"] for r in rows)/1e6,
            median_gpu_buffer_mb=statistics.median(r["gpu_buffer_bytes"] for r in rows)/1e6))
    full_runs = []
    full_comparisons = []
    full_fingerprints = {}
    for path in (args.root / "full").glob("*.json"):
        if path.name.endswith(".steps.json"):
            continue
        row = json.loads(path.read_text())
        if not row["native"]:
            errors.append(str(path) + ": full-solve process failed")
            continue
        report = row["native"]["report"]
        if report["status"] == "sat":
            fingerprint = (digest(path.with_suffix(".assignment.bin")),
                           digest(path.with_suffix(".steps.json")))
            key = (Path(row["job"]["formula"]).name, row["job"]["seed"],
                   row["job"]["backend"])
            expected = full_fingerprints.setdefault(key, fingerprint)
            if fingerprint != expected or row["independent_verified"] is not True:
                errors.append(str(path) + ": full-solve group failed witness/trajectory check")
        full_runs.append(dict(name=path.stem, variant=Path(row["job"]["binary"]).name,
            backend=row["job"]["backend"], status=report["status"],
            seconds=report["solve_seconds"], verified=row["independent_verified"],
            footprint_mb=row["observed_lifetime_peak_phys_footprint_bytes"]/1e6,
            gpu_buffer_mb=report["execution"]["allocated_bytes"]/1e6
                if row["job"]["backend"] == "metal" else 0))
        if "reference" not in row["job"]:
            continue
        reference = args.root / "full" / Path(row["job"]["reference"]).name
        if not reference.exists():
            continue
        base = json.loads(reference.read_text())
        a = base["native"]["report"]
        if a["status"] != "sat" or report["status"] != "sat":
            # Keep censored runs visible, without a solution-time ratio.
            continue
        matched = (row["independent_verified"] is True and
            base["independent_verified"] is True and
            digest(path.with_suffix(".assignment.bin")) == digest(reference.with_suffix(".assignment.bin")) and
            digest(path.with_suffix(".steps.json")) == digest(reference.with_suffix(".steps.json")))
        if not matched:
            errors.append(str(path) + ": full witness or trajectory mismatch")
        full_comparisons.append(dict(name=path.stem, reference=reference.stem,
            matched=matched, speedup=a["solve_seconds"]/report["solve_seconds"]))
    args.output.write_text(json.dumps(dict(summary=summary, comparisons=comparisons,
        full_runs=full_runs, full_comparisons=full_comparisons, mismatches=errors), indent=2) + "\n")
    print(f"{len(comparisons)} fixed-work and {len(full_comparisons)} full-solve paired controls, {len(errors)} mismatches")
    if errors:
        raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
