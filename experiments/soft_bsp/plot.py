"""Render a paired complexity plot with gnuplot 6."""

import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    policies = [
        "bias_soft", "immediate_hard", "immediate_soft", "lookahead_hard", "lookahead_soft",
    ]
    labels = ["Bias soft", "Immediate hard", "Immediate soft", "Lookahead hard", "Lookahead soft"]
    script = [
        "set terminal pngcairo size 1200,750 font 'Helvetica,14'",
        f"set output {json.dumps(str(args.output))}",
        "set xrange [-0.031:0.009]",
        "set xtics -0.03,0.01,0",
        "set format x '%.2f'",
        "set yrange [5.6:0.4]",
        "set xlabel 'Paired difference in total complexity'",
        "set ytics (" + ", ".join(f'"{label}" {i}' for i, label in enumerate(labels, 1)) + ")",
        "set lmargin 17",
        "set grid xtics lc rgb '#e5e7eb'",
        "set xzeroaxis lw 1.5 lc rgb '#6b7280'",
        "unset key",
    ]
    panels = ["set multiplot layout 2,2 title 'One-move SP complexity relative to native BSP'"]
    for alpha, group in sorted(summary.items()):
        for phase, color in [("fix", "#0072b2"), ("release", "#d55e00")]:
            result = group["checkpoints"][phase]
            data = []
            for i, name in enumerate(policies, 1):
                effect = result["policies"][name]["immediate"]
                low, high = effect["bootstrap95"]
                data.append(f"{effect['mean']} {i} {low} {high}")
            block = f"${phase}{alpha.replace('.', '')}"
            script.extend([f"{block} << EOD", *data, "EOD"])
            panels.extend([
                f"set title '{phase.capitalize()}, alpha={alpha} ({result['eligible']} checkpoints)'",
                f"plot {block} using 1:2:3:4 with xerrorbars pt 7 ps 1.2 lw 2 lc rgb '{color}'",
            ])
    panels.append("unset multiplot")
    subprocess.run(["gnuplot"], input="\n".join(script + panels), text=True, check=True)


if __name__ == "__main__":
    main()
