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
    for kind in ("actions", "kernels"):
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
            matched = (a["status"] == b["status"] == "ok" and
                       a["message_hash"] == b["message_hash"])
            if kind == "actions":
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
    args.output.write_text(json.dumps(dict(summary=summary, comparisons=comparisons,
        mismatches=errors), indent=2) + "\n")
    print(f"{len(comparisons)} paired controls, {len(errors)} mismatches")
    if errors:
        raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
