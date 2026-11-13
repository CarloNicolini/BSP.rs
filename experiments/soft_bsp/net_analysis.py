"""Verify and analyze net-depth BSP runs with paired formula-level intervals."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean, median

from analyze import complexity, paired_summary

ARMS = ("native", "immediate", "net_hard", "net_soft")


def verify_witness(formula, n, run):
    witness = run["witness"]
    assert (witness is not None) == (run["status"] == "sat")
    if witness is None:
        return
    assert len(witness) == n and set(witness) <= {0, 1}
    for line in formula.read_text().splitlines():
        if line and line[0] not in "cp":
            clause = map(int, line.split()[:-1])
            assert any(witness[abs(lit) - 1] == int(lit > 0) for lit in clause)


def read_runs(directory, mode):
    seeds = range(211, 215) if mode == "development" else range(301, 313)
    results = defaultdict(dict)
    for n in (640, 1280):
        for seed in seeds:
            formula = directory / f"formula-{n}-{seed}.cnf"
            assert formula.read_text().splitlines()[0].split() == [
                "p", "cnf", str(n), str(int(9.65 * n))
            ]
            for arm in ARMS:
                data = json.loads((directory / f"case-{n}-{seed}-{arm}.json").read_text())
                assert (data["mode"], data["n"], data["seed"]) == (mode, n, seed)
                run = data["result"]
                assert run["arm"] == arm
                verify_witness(formula, n, run)
                results[n][seed, arm] = run
    return results


def paired_effect(probabilities, values):
    if values[0] is None or any(p > 0 and x is None for p, x in zip(probabilities, values)):
        return None
    return sum(p * (x - values[0]) for p, x in zip(probabilities, values) if p > 0)


def summarize_checkpoints(group, phase):
    metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    endpoints = Counter()
    complete = count = undo = attempted = 0
    baseline = defaultdict(list)
    costs = []
    net_spreads = []
    net_moves = []
    contrasts = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for (seed, arm), run in group.items():
        if arm != "native":
            continue
        for point in run["checkpoints"]:
            if point["phase"] != phase:
                continue
            count += 1
            before = point["before"]["complexity"]
            first = [complexity(branch["first"]) for branch in point["branches"]]
            follow = [branch["continuation"] for branch in point["branches"]]
            depth = [branch["complexity"] if branch else None for branch in follow]
            complete += all(x is not None for x in depth)
            costs.append(point["sweeps"])
            if None not in depth:
                net_spreads.append(max(depth) - min(depth))
            for branch in follow:
                end = branch["end"] if branch else "missing_first"
                endpoints[end if isinstance(end, str) else end["status"]] += 1
                if branch:
                    attempted += 1
                    undo += branch["undo"]
                    if branch["end"] == "reached":
                        net_moves.append(branch["moves"])
            for label, values in (("immediate", first), ("net_depth", depth)):
                if values[0] is not None:
                    baseline[label].append(values[0] - before)
            point_effects = {}
            for policy in point["policies"]:
                weights = policy["probabilities"]
                assert abs(sum(weights) - 1) < 1e-12
                for label, values in (("immediate", first), ("net_depth", depth)):
                    value = paired_effect(weights, values)
                    if value is not None:
                        metrics[policy["arm"]][label][seed].append(value)
                        point_effects[policy["arm"], label] = value
            for label in ("immediate", "net_depth"):
                for treatment, control in (("net_hard", "immediate"), ("net_soft", "net_hard")):
                    a, b = point_effects.get((treatment, label)), point_effects.get((control, label))
                    if a is not None and b is not None:
                        contrasts[f"{treatment} minus {control}"][label][seed].append(a - b)
    return {
        "checkpoints": count,
        "complete_net_checkpoints": complete,
        "continuation_statuses": endpoints,
        "immediate_undo": [undo, attempted],
        "median_probe_sweeps": median(costs) if costs else None,
        "median_followup_moves": median(net_moves) if net_moves else None,
        "max_followup_moves": max(net_moves, default=None),
        "net_spread_below_1e_6": [sum(v < 1e-6 for v in net_spreads), len(net_spreads)],
        "paired_contrasts": {
            name: {label: paired_summary(values) for label, values in labels.items()}
            for name, labels in contrasts.items()
        },
        "native_mean_delta": {label: mean(values) for label, values in baseline.items()},
        "relative_to_native": {
            arm: {label: paired_summary(arms.get(label, {})) for label in ("immediate", "net_depth")}
            for arm, arms in metrics.items()
        },
    }


def summarize(group):
    seeds = sorted({seed for seed, _ in group})
    return {
        "formulas": len(seeds),
        "closed_loop": {
            arm: {
                "status": Counter(run["status"] for run in runs),
                "budget_stops": Counter(run["stop_budget"] for run in runs if run["stop_budget"]),
                "median_sweeps": median(run["sweeps"] for run in runs),
                "max_sweeps": max(run["sweeps"] for run in runs),
                "median_wall_seconds": median(run["wall_seconds"] for run in runs),
                "median_speculative_sweeps": median(run["speculative_sweeps"] for run in runs),
                "median_high_water_decisions": median(run["high_water_decisions"] for run in runs),
                "discordant_sat": {
                    "only_arm": sum(a["status"] == "sat" and b["status"] != "sat" for a, b in pairs),
                    "only_native": sum(a["status"] != "sat" and b["status"] == "sat" for a, b in pairs),
                },
            }
            for arm in ARMS
            for runs in [[group[seed, arm] for seed in seeds]]
            for pairs in [list(zip(runs, (group[seed, "native"] for seed in seeds)))]
        },
        "checkpoints": {phase: summarize_checkpoints(group, phase) for phase in ("fix", "release")},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("development", "heldout"))
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    data = read_runs(args.directory, args.mode)
    print(json.dumps({n: summarize(group) for n, group in data.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
