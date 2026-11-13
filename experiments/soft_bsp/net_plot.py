"""Plot the held-out complexity effects at equal net decision depth."""
import argparse
import json
from pathlib import Path
import subprocess

from net_analysis import ARMS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    blocks, panels, extent = [], [], 0.0
    labels = ["Immediate hard", "Net-depth hard", "Net-depth soft"]
    for n, group in sorted(summary.items(), key=lambda item: int(item[0])):
        for phase, color in (("fix", "#0072b2"), ("release", "#d55e00")):
            result = group["checkpoints"][phase]
            block = f"${phase}{n}"
            data = []
            for y, arm in enumerate(ARMS[1:], 1):
                effect = result["relative_to_native"][arm]["net_depth"]
                low, high = effect["bootstrap95"]
                extent = max(extent, abs(low), abs(high))
                data.append(f"{effect['mean']} {y} {low} {high}")
            blocks.extend([f"{block} << EOD", *data, "EOD"])
            panels.extend([
                f"set title 'N={n}, {phase} ({result['complete_net_checkpoints']} complete checkpoints)'",
                f"plot {block} using 1:2:3:4 with xerrorbars pt 7 ps 1.2 lw 2 lc rgb '{color}'",
            ])
    setup = [
        "set terminal pngcairo size 1200,690 font 'Helvetica,14'",
        f"set output {json.dumps(str(args.output))}",
        "set multiplot layout 2,2 title 'SP complexity at three additional net decisions'",
        f"set xrange [-{max(0.0001, extent * 1.1)}:{max(0.0001, extent * 1.1)}]",
        "set xtics 0.001",
        "set format x '%.4f'",
        "set yrange [3.6:0.4]",
        "set xlabel 'Paired difference from native BSP'",
        "set ytics (" + ", ".join(f'"{label}" {i}' for i, label in enumerate(labels, 1)) + ")",
        "set lmargin 18",
        "set grid xtics lc rgb '#e5e7eb'",
        "set xzeroaxis lw 1.5 lc rgb '#6b7280'",
        "unset key",
    ]
    subprocess.run(["gnuplot"], input="\n".join(blocks + setup + panels + ["unset multiplot"]), text=True, check=True)


if __name__ == "__main__":
    main()
