"""Run or resume the frozen staged soft-BSP experiment in one process at a time."""
import argparse
from collections import Counter, defaultdict
import fcntl
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
from statistics import mean, median
import subprocess

from net_analysis import verify_witness
from staged_fit import assess, train

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SIZES = (640, 1280)
GRID = (9.25, 9.35, 9.45, 9.55)
ARMS = ("native", "bias_soft", "net_soft", "predictive")


def write(path, data, frozen=False):
    text = json.dumps(data, indent=2, sort_keys=True) + "\n"
    if frozen and path.exists():
        if path.read_text() != text:
            raise ValueError(f"Frozen file differs: {path}")
        return
    pending = path.with_suffix(".tmp")
    with pending.open("w") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)


def build():
    completed = subprocess.run(
        ["cargo", "test", "--locked", "--release", "--lib", "--no-run", "--message-format=json"],
        cwd=REPO, text=True, stdout=subprocess.PIPE, check=True,
    )
    for line in completed.stdout.splitlines():
        message = json.loads(line)
        if message.get("executable") and message["target"]["name"] == "bsp_rust":
            return message["executable"]
    raise RuntimeError("Cargo did not produce the BSP library test executable")


class Campaign:
    def __init__(self, root, binary):
        self.root, self.binary = root, binary
        for name in ("formulas", "runs", "manifests"):
            (root / name).mkdir(parents=True, exist_ok=True)

    def jobs(self, stage, grids, arms):
        start, count = {
            "calibrate": (10_000, 16), "extend": (20_000, 16),
            "train": (30_000, 24), "evaluate": (40_000, 24),
            "explore": (40_000, 24), "confirm": (50_000, 160),
        }[stage]
        entries = []
        for index, (n, alpha) in enumerate(grids):
            seed_base = start + (SIZES.index(n) if stage not in ("calibrate", "extend") else index) * 1000
            for seed in range(seed_base, seed_base + count):
                order = list(arms)
                random.Random(seed ^ 0x5B50).shuffle(order)
                stem = f"{n}-{alpha:.4f}-{seed}"
                for arm in order:
                    entries.append({
                        "stage": stage, "n": n, "alpha": alpha, "formula_seed": seed,
                        "solver_seed": 23, "policy_seed": seed + 100_000, "arm": arm,
                        "formula": str(self.root / "formulas" / f"{stem}.cnf"),
                        "output": str(self.root / "runs" / f"{stage}-{stem}-{arm}.json"),
                    })
        return entries

    def execute(self, stage, jobs, model=False):
        manifest = self.root / "manifests" / f"{stage}.jsonl"
        payload = "".join(json.dumps(j, sort_keys=True) + "\n" for j in jobs)
        if manifest.exists() and manifest.read_text() != payload:
            raise ValueError(f"Frozen job manifest differs: {manifest}")
        if not manifest.exists():
            pending = manifest.with_suffix(".tmp")
            pending.write_text(payload)
            pending.replace(manifest)
        env = os.environ.copy()
        env.pop("SOFT_BSP_MODEL", None)
        env["SOFT_BSP_JOBS"] = str(manifest)
        if model:
            env["SOFT_BSP_MODEL"] = str(self.root / "policy-model.json")
        subprocess.run(
            [
                self.binary, "solver::soft_policy_experiment::staged::staged_bsp_campaign",
                "--ignored", "--exact", "--nocapture", "--test-threads=1"
            ], cwd=REPO, env=env, check=True,
        )
        records = []
        for job in jobs:
            record = json.loads(Path(job["output"]).read_text())
            if record["job"] != job:
                raise ValueError(f"Result belongs to another job: {job['output']}")
            formula = Path(job["formula"])
            verify_witness(formula, job["n"], record["result"])
            for point in record["checkpoints"]:
                for branch in point["branches"]:
                    for label in branch["labels"]:
                        verify_witness(formula, job["n"], label)
            records.append(record)
        return records

    def calibrate(self):
        grid = list(itertools.product(SIZES, GRID))
        records = self.execute("calibrate", self.jobs("calibrate", grid, ["native"]))
        counts = calibration_counts(records)
        extra = []
        for n in SIZES:
            if qualifying(counts[n]) is None:
                extra.append((n, extension(counts[n])))
        if extra:
            records += self.execute("extend", self.jobs("extend", extra, ["native"]))
            counts = calibration_counts(records)
        chosen = [(n, alpha) for n in SIZES if (alpha := qualifying(counts[n])) is not None]
        write(self.root / "calibration.json", {"counts": counts, "selected": chosen}, frozen=True)
        return chosen


def calibration_counts(records):
    counts = defaultdict(lambda: defaultdict(lambda: {"sat": 0, "total": 0}))
    for record in records:
        job = record["job"]
        bucket = counts[job["n"]][job["alpha"]]
        bucket["total"] += 1
        bucket["sat"] += record["result"]["status"] == "sat"
    return counts


def qualifying(counts):
    eligible = [
        (abs(v["sat"] / v["total"] - 0.5), -alpha)
        for alpha, v in counts.items() if 0.3 <= v["sat"] / v["total"] <= 0.7
    ]
    return -min(eligible)[1] if eligible else None


def extension(counts):
    values = sorted((a, v["sat"] / v["total"]) for a, v in counts.items())
    pairs = [
        (b - a, -(a + b) / 2) for (a, p), (b, q) in zip(values, values[1:])
        if (p - 0.5) * (q - 0.5) < 0
    ]
    if pairs:
        return round(-min(pairs)[1], 4)
    return round(values[0][0] - 0.1 if values[0][1] < 0.5 else values[-1][0] + 0.1, 4)


def pairs(records, left, right):
    formulas = defaultdict(dict)
    for record in records:
        job = record["job"]
        formulas[job["n"], job["formula_seed"]][job["arm"]] = record["result"]["status"] == "sat"
    by_n = defaultdict(list)
    for (n, _), runs in sorted(formulas.items()):
        by_n[n].append(int(runs[left]) - int(runs[right]))
    return by_n


def contrast(records, left, right):
    groups = pairs(records, left, right)
    flattened = list(itertools.chain.from_iterable(groups.values()))
    win, loss = flattened.count(1), flattened.count(-1)
    discordant = win + loss
    p = min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(win, loss) + 1)) / 2**discordant)
    rng = random.Random(6977)
    draws = sorted(
        sum(sum(rng.choices(group, k=len(group))) for group in groups.values()) / len(flattened)
        for _ in range(20_000)
    )
    effect = mean(flattened)
    low, high = draws[250], draws[19_749]
    return {
        "difference": effect, "wins": win, "losses": loss, "pairs": len(flattened),
        "interval_97_5": [low, high], "mcnemar_p": p,
        "practical_and_significant": effect >= 0.10 and low > 0 and p <= 0.025,
        "by_n": {n: dict(Counter(values)) for n, values in groups.items()},
    }


def summarize(records):
    groups = defaultdict(list)
    for record in records:
        job = record["job"]
        groups[job["n"], job["arm"]].append(record)
    return {
        f"{n}/{arm}": {
            "status": Counter(r["result"]["status"] for r in rows),
            "budget_stops": Counter(r["result"]["stop"] for r in rows if r["result"]["stop"]),
            "median_sweeps": median(r["result"]["sweeps"] for r in rows),
            "median_seconds": median(r["result"]["seconds"] for r in rows),
            "median_speculative_sweeps": median(r["speculative_sweeps"] for r in rows),
            "success_curve": {
                metric: {
                    cut: sum(r["result"]["status"] == "sat" and r["result"][metric] <= cut for r in rows)
                    for cut in cuts
                }
                for metric, cuts in (
                    ("seconds", (15, 30, 45, 60, 90, 120, 180)),
                    ("sweeps", (50_000, 100_000, 250_000, 500_000, 1_000_000, 1_500_000)),
                )
            },
        }
        for (n, arm), rows in sorted(groups.items())
    }


def gate(records):
    wins = Counter(r["job"]["arm"] for r in records if r["result"]["status"] == "sat")
    elapsed = {
        arm: sum(r["result"]["seconds"] for r in records if r["job"]["arm"] == arm)
        for arm in ARMS
    }
    eligible = [
        arm for arm in ("net_soft", "predictive")
        if wins[arm] >= max(wins["native"], wins["bias_soft"]) + 2
    ]
    selected = min(eligible, key=lambda arm: (-wins[arm], elapsed[arm], arm != "predictive"), default=None)
    bias = wins["bias_soft"] >= wins["native"] + 2
    return {"bias_advances": bias, "continuation": selected, "sat_counts": dict(wins)}


def provenance(binary):
    files = [
        HERE / "staged.py", HERE / "staged_fit.py", HERE / "staged_driver.rs",
        HERE / "driver.rs", HERE / "STAGED_PROTOCOL.md", REPO / "Cargo.lock",
    ]
    return {
        "baseline": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "rustc": subprocess.check_output(["rustc", "-Vv"], text=True),
        "system": [os.uname().sysname, os.uname().release, os.uname().machine],
        "binary_sha256": hashlib.sha256(Path(binary).read_bytes()).hexdigest(),
        "inputs": {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    }


def run_campaign(root, stage):
    binary = build()
    snapshot = provenance(binary)
    write(root / "provenance.json", snapshot, frozen=True)
    campaign = Campaign(root, binary)
    chosen = campaign.calibrate()
    print("Calibrated regimes:", chosen, flush=True)
    if stage == "calibrate" or not chosen:
        write(root / "completed.json", {"phase": "calibration", "qualified": chosen})
        return
    development = campaign.execute("train", campaign.jobs("train", chosen, ["native"]))
    models = train(development)
    write(root / "models.json", models, frozen=True)
    write(root / "policy-model.json", models["full"], frozen=True)
    pilot = campaign.execute("explore", campaign.jobs("explore", chosen, ARMS), model=True)
    contrasts = {
        f"{left}/{right}": contrast(pilot, left, right)
        for left, right in (("bias_soft", "native"), ("net_soft", "bias_soft"), ("predictive", "bias_soft"),
                            ("net_soft", "native"), ("predictive", "native"))
    }
    decision = gate(pilot)
    write(root / "exploration.json", {"runs": summarize(pilot), "contrasts": contrasts, "gate": decision}, frozen=True)
    print("Exploration gate:", decision, flush=True)
    if decision["bias_advances"] or decision["continuation"]:
        arms = ["native", "bias_soft"]
        if decision["continuation"]:
            arms.append(decision["continuation"])
        confirmation = campaign.execute("confirm", campaign.jobs("confirm", chosen, arms), model=True)
        tests = {"bias_soft/native": contrast(confirmation, "bias_soft", "native")}
        if len(arms) == 3:
            tests[f"{arms[2]}/bias_soft"] = contrast(confirmation, arms[2], "bias_soft")
        write(root / "confirmation.json", {"runs": summarize(confirmation), "contrasts": tests}, frozen=True)
    evaluation = campaign.execute("evaluate", campaign.jobs("evaluate", chosen, ["native"]))
    write(root / "predictor-evaluation.json", assess(evaluation, models), frozen=True)
    write(root / "completed.json", {"phase": "full", "gate": decision})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--stage", choices=("calibrate", "all"), default="all")
    args = parser.parse_args()
    root = args.directory.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "campaign.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_campaign(root, args.stage)


if __name__ == "__main__":
    main()
