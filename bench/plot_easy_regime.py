"""Render the README's measured Apple M1 Pro speedups; requires matplotlib.

Run from any directory: python /path/to/BSP.rs/bench/plot_easy_regime.py
The CSV contains time to one verified solution, including the corrected C++
one-output build with whitening disabled. CUDA is unmeasured and is omitted.
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "bench/data/apple_m1_pro_alpha4.csv"
OUTPUT = ROOT / "docs/_static/benchmarks/apple-m1-pro-alpha4.svg"

with DATA.open(newline="") as handle:
    rows = list(csv.DictReader(handle))

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "svg.fonttype": "none", "svg.hashsalt": "bsp-m1-pro-alpha4"})
fig, ax = plt.subplots(figsize=(9, 4.8), facecolor="white")
fig.subplots_adjust(left=.09, right=.98, bottom=.16, top=.76)
fig.text(.09, .95, "Time to one verified solution: speedup over C++",
         fontsize=14, weight="bold", color="#1f2937")
fig.text(.09, .90, "Apple M1 Pro, 16 GiB  |  random 3-SAT, alpha = 4  |  Python APIs",
         fontsize=10, color="#4b5563")

positions = list(range(len(rows)))
width = .34
for key, label, color, offset in [
    ("cpu_seconds", "Rust CPU (f64)", "#0072B2", -width / 2),
    ("metal_seconds", "Rust Metal (f32)", "#009E73", width / 2),
]:
    values = [float(r["cpp_seconds"]) / float(r[key]) for r in rows]
    bars = ax.bar([x + offset for x in positions], values, width, color=color, label=label, zorder=3)
    ax.bar_label(bars, labels=[f"{v:.2f}x" for v in values], padding=4, fontsize=9,
                 color="#1f2937", bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})

ax.axhline(1, color="#6b7280", linestyle=(0, (4, 3)), linewidth=1.1, zorder=2)
handles, labels = ax.get_legend_handles_labels()
handles.append(Line2D([], [], color="#6b7280", linestyle="--", linewidth=1.1))
labels.append("C++ reference (1x)")
fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.08, .865), ncol=3,
           frameon=False, fontsize=10)
ax.set_xticks(positions, [f"{int(r['n']):,}" for r in rows])
ax.set_xlabel("Variables")
ax.set_ylabel("Speedup (x, higher is faster)")
ax.set_ylim(0, 14)
ax.set_yticks(range(0, 15, 2))
ax.set_axisbelow(True)
ax.grid(axis="y", color="#e5e7eb", linewidth=.7)
ax.spines[["top", "right"]].set_visible(False)
for side in ["left", "bottom"]:
    ax.spines[side].set_color("#9ca3af")
fig.text(.09, .035, "C++: one output, whitening disabled. Three formulas per size; medians across repeats, then mean.",
         fontsize=8, color="#4b5563")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT, format="svg", metadata={"Date": None})
svg = OUTPUT.read_text()
start = svg.index("<svg ")
end = svg.index(">", start) + 1
svg = svg[:start] + svg[start:end].replace("<svg ", '<svg role="img" aria-labelledby="chart-title chart-description" ', 1) + (
    '\n<title id="chart-title">Measured speedup over C++ on Apple M1 Pro</title>'
    '\n<desc id="chart-description">One verified solution per call, random 3-SAT at alpha 4. '
    'Six formula sizes from 500 to 20000 variables. CPU is fastest at small sizes; '
    'Metal reaches 12.68 times the C++ speed at 20000 variables. CUDA is unmeasured. '
    'The C++ benchmark requests one solution with whitening disabled.</desc>'
) + svg[end:]
OUTPUT.write_text("\n".join(line.rstrip() for line in svg.splitlines()) + "\n")
plt.close(fig)
print(OUTPUT)
