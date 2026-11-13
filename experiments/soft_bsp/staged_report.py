"""Report completed staged experiments and optionally plot success versus time."""
import argparse
import json
from pathlib import Path
from collections import Counter
import subprocess


def read(directory, name):
    path = directory / (name + ".json")
    return json.loads(path.read_text()) if path.exists() else None


def progress(directory):
    for path in sorted((directory / "manifests").glob("*.jsonl")):
        jobs = [json.loads(line) for line in path.read_text().splitlines()]
        paths = [directory / "runs" / Path(job["output"]).name for job in jobs]
        counts = Counter(json.loads(p.read_text())["result"]["status"] for p in paths if p.exists())
        print(f"{path.stem}: {sum(counts.values())}/{len(jobs)} runs {dict(counts)}")


def report(directory):
    progress(directory)
    calibration = read(directory, "calibration")
    if calibration:
        print("\nSelected calibration regimes:", calibration["selected"])
    for stage in ("exploration", "confirmation"):
        data = read(directory, stage)
        if data is None:
            continue
        print(f"\n{stage.capitalize()}")
        for group, stats in data["runs"].items():
            successes = stats["status"].get("sat", 0)
            total = sum(stats["status"].values())
            print(f"- {group} {successes}/{total} SAT, median {stats['median_seconds']:.2f} s,"
                  f" {stats['median_speculative_sweeps']:.0f} speculative sweeps")
        for pair, stats in data["contrasts"].items():
            low, high = stats["interval_97_5"]
            print(f"- {pair} {stats['difference']:+.1%} [{low:+.1%}, {high:+.1%}],"
                  f" {stats['wins']} wins and {stats['losses']} losses")
        if "gate" in data:
            print("Gate:", data["gate"])
    evaluation = read(directory, "predictor-evaluation")
    if evaluation:
        print("\nPredictor evaluation")
        for group, stats in evaluation.items():
            print(f"- {group} {stats['complete_features']}/{stats['checkpoints']} complete checkpoints,"
                  f" {stats['label_ties']} ties, Brier bias/depth {stats['bias_depth']['brier']},"
                  f" full {stats['full']['brier']}")


def plot(directory, output):
    stage = "confirmation" if read(directory, "confirmation") else "exploration"
    data = read(directory, stage)
    if not data:
        raise ValueError("A complete exploration or confirmation stage is required to plot")
    by_n = {}
    for group, stats in data["runs"].items():
        n, arm = group.split("/")
        by_n.setdefault(n, {})[arm] = stats
    blocks, panels = [], []
    for n, arms in sorted(by_n.items(), key=lambda item: int(item[0])):
        panels.extend([f"set title 'N={n}'", f"set xrange [0:{90 if n == '640' else 180}]"])
        plots = []
        for arm, stats in sorted(arms.items()):
            key = f"${n}_{arm}"
            total = sum(stats["status"].values())
            points = sorted((int(t), value / total) for t, value in stats["success_curve"]["seconds"].items())
            blocks.extend([f"{key} << EOD", "0 0", *(f"{t} {p}" for t, p in points), "EOD"])
            plots.append(f"{key} with linespoints title '{arm.replace('_', ' ')}'")
        panels.append("plot " + ", ".join(plots))
    setup = [
        "set terminal pngcairo size 1200,500 font 'Helvetica,14'",
        f"set output {json.dumps(str(output))}",
        f"set multiplot layout 1,{len(by_n)} title '{stage.capitalize()} verified SAT'",
        "set xlabel 'Wall time (seconds)'",
        "set ylabel 'Fraction verified SAT'",
        "set yrange [0:1]",
        "set grid ytics",
        "set key bottom right",
    ]
    subprocess.run(["gnuplot"], input="\n".join(blocks + setup + panels + ["unset multiplot"]),
                   text=True, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    report(args.directory)
    if args.plot:
        plot(args.directory, args.plot)


if __name__ == "__main__":
    main()
