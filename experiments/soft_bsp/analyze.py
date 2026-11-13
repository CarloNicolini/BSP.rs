"""Verify SAT witnesses and summarize paired counterfactuals from cases.jsonl."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
from statistics import mean, median


def read_cases(directory):
    records = [json.loads(line) for line in (directory / "cases.jsonl").read_text().splitlines()]
    cases = sorted(records, key=lambda case: (case["alpha"], case["seed"]))
    if len({(case["alpha"], case["seed"]) for case in cases}) != len(cases):
        raise ValueError("Duplicate formula seed at a density")
    return cases


def verify_witnesses(directory, case):
    path = directory / f"formula-{case['alpha']}-{case['seed']}.cnf"
    clauses = [
        [int(lit) for lit in line.split()[:-1]]
        for line in path.read_text().splitlines()
        if line and line[0] not in "cp"
    ]
    for run in case["policies"]:
        witness = run["witness"]
        assert (witness is not None) == (run["status"] == "sat")
        if witness is not None:
            assert len(witness) == case["n"] and set(witness) <= {0, 1}
            assert all(any(witness[abs(lit) - 1] == (lit > 0) for lit in c) for c in clauses)


def complexity(result):
    if result and "Ok" in result:
        return result["Ok"]["complexity"]
    return None


def change(outcome, before, second):
    first = complexity(outcome["sp"])
    if first is None:
        return None
    followup = outcome["native_second"]
    if second and followup is not None:
        value = complexity(followup)
        return None if value is None else value - before
    return first - before


def expected_difference(probabilities, outcomes, before, second):
    deltas = [change(outcome, before, second) for outcome in outcomes]
    if deltas[0] is None or any(p > 0 and delta is None for p, delta in zip(probabilities, deltas)):
        return None
    return sum(p * (delta - deltas[0]) for p, delta in zip(probabilities, deltas) if p > 0)


def paired_summary(by_seed):
    """Average within formula before resampling formulas with replacement."""
    values = [mean(deltas) for _, deltas in sorted(by_seed.items())]
    if not values:
        return {"formulas": 0, "checkpoints": 0, "mean": None, "bootstrap95": None}
    rng = random.Random(697)
    draws = sorted(mean(rng.choices(values, k=len(values))) for _ in range(5000))
    return {
        "formulas": len(values),
        "checkpoints": sum(map(len, by_seed.values())),
        "mean": mean(values),
        "bootstrap95": [draws[125], draws[4874]] if len(values) > 1 else None,
    }


def analyze(directory, cases):
    result = {}
    for alpha in sorted({case["alpha"] for case in cases}):
        group = [case for case in cases if case["alpha"] == alpha]
        checkpoints = defaultdict(list)
        runs = defaultdict(list)
        for case in group:
            verify_witnesses(directory, case)
            for run in case["policies"]:
                runs[run["policy"]["name"]].append(run)
            for checkpoint in case["policies"][0]["checkpoints"]:
                checkpoints[checkpoint["phase"]].append((case["seed"], checkpoint))
        result[alpha] = {
            "formulas": len(group),
            "trajectories": {
                name: {
                    "statuses": Counter(run["status"] for run in series),
                    "median_sweeps": median(run["sweeps"] for run in series),
                    "median_wall_seconds": median(run["wall_seconds"] for run in series),
                    "trajectory_delta": {
                        phase: mean(
                            run["deltas"][i]["sum"] / run["deltas"][i]["count"]
                            for run in series if run["deltas"][i]["count"]
                        ) if any(run["deltas"][i]["count"] for run in series) else None
                        for i, phase in enumerate(("fix", "release"))
                    },
                }
                for name, series in runs.items()
            },
            "checkpoints": {
                phase: summarize_checkpoints(records, len(group))
                for phase, records in checkpoints.items()
            },
        }
    return result


def summarize_checkpoints(records, formulas):
    policies = defaultdict(lambda: [defaultdict(list), defaultdict(list)])
    failures = defaultdict(lambda: [[], []])
    selected_other = defaultdict(list)
    contrasts = defaultdict(lambda: [defaultdict(list), defaultdict(list)])
    baseline = []
    children = Counter()
    oracle_root = Counter()
    sat_to_unsat = 0
    probe_sweeps = []
    for seed, checkpoint in records:
        assert checkpoint["zero_limit_matches"]
        outcomes = checkpoint["outcomes"]
        before = checkpoint["before"]["complexity"]
        oracle_root[checkpoint["oracle"]] += 1
        children.update(outcome["oracle"] for outcome in outcomes)
        sat_to_unsat += sum(
            checkpoint["oracle"] == "sat" and outcome["oracle"] == "unsat"
            for outcome in outcomes
        )
        probe_sweeps.append(checkpoint["probe_sweeps"])
        baseline.append(change(outcomes[0], before, False))
        checkpoint_effects = {}
        for policy in checkpoint["policies"]:
            name = policy["name"]
            if name == "sparse_lookahead_hard":
                continue  # Same scoring as lookahead_hard at an evaluated root.
            probabilities = policy["probabilities"]
            assert len(probabilities) == len(outcomes) and abs(sum(probabilities) - 1) < 1e-10
            selected_other[name].append(1 - probabilities[0])
            checkpoint_effects[name] = []
            for second in (False, True):
                delta = expected_difference(probabilities, outcomes, before, second)
                checkpoint_effects[name].append(delta)
                if delta is not None:
                    policies[name][second][seed].append(delta)
                failures[name][second].append(sum(
                    p for p, outcome in zip(probabilities, outcomes)
                    if change(outcome, before, second) is None
                ))
        for treatment, control in [
            ("immediate_soft", "immediate_hard"),
            ("lookahead_soft", "lookahead_hard"),
            ("lookahead_hard", "immediate_hard"),
            ("lookahead_soft", "immediate_soft"),
        ]:
            for horizon, (a, b) in enumerate(
                zip(checkpoint_effects[treatment], checkpoint_effects[control])
            ):
                if a is not None and b is not None:
                    contrasts[f"{treatment} minus {control}"][horizon][seed].append(a - b)
    valid = [value for value in baseline if value is not None]
    return {
        "eligible": len(records),
        "planned": 2 * formulas,
        "baseline_valid": len(valid),
        "baseline_mean_delta": mean(valid) if valid else None,
        "median_probe_sweeps": median(probe_sweeps),
        "root_oracle": oracle_root,
        "child_oracle": children,
        "sat_to_unsat": sat_to_unsat,
        "paired_contrasts": {
            name: {
                "immediate": paired_summary(estimates[0]),
                "two_steps_native_followup": paired_summary(estimates[1]),
            }
            for name, estimates in contrasts.items()
        },
        "policies": {
            name: {
                "immediate": paired_summary(policies[name][0]),
                "two_steps_native_followup": paired_summary(policies[name][1]),
                "missing_mass_immediate": mean(failures[name][0]),
                "missing_mass_two_steps": mean(failures[name][1]),
                "probability_of_non_native_move": mean(selected_other[name]),
            }
            for name in selected_other
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--expected", type=int, required=True)
    args = parser.parse_args()
    cases = read_cases(args.directory)
    if len(cases) != args.expected:
        parser.error(f"expected {args.expected} completed cases, found {len(cases)}")
    print(json.dumps(analyze(args.directory, cases), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
