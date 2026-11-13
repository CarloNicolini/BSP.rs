"""Fit cheap continuation predictors on the pilot and score them on held-out cases."""

import argparse
from collections import Counter
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean

from analyze import complexity, paired_summary, read_cases


def examples(cases):
    groups = defaultdict(list)
    for case in cases:
        for checkpoint in case["policies"][0]["checkpoints"]:
            before = checkpoint["before"]["complexity"]
            for branch in checkpoint["outcomes"]:
                first = complexity(branch["sp"])
                second = complexity(branch["native_second"])
                if first is not None and second is not None:
                    groups[(case["alpha"], checkpoint["phase"])].append(
                        (case["seed"], first - before, second - first)
                    )
    return groups


def fit(rows):
    # One-feature least squares with an intercept. Pairing and grouping are
    # by formula at evaluation; no holdout rows enter this calculation.
    xbar = mean(row[1] for row in rows)
    ybar = mean(row[2] for row in rows)
    variance = sum((x - xbar) ** 2 for _, x, _ in rows)
    covariance = sum((x - xbar) * (y - ybar) for _, x, y in rows)
    return xbar, ybar, covariance / variance if variance else 0.0


def mean_absolute_error(rows, predict):
    errors = defaultdict(list)
    for seed, x, actual in rows:
        errors[seed].append(abs(predict(x) - actual))
    return mean(map(mean, errors.values()))


def diagnostic(cases):
    by_group = defaultdict(Counter)
    errors = defaultdict(lambda: defaultdict(list))
    transfers = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for case in cases:
        for checkpoint in case["policies"][0]["checkpoints"]:
            before = checkpoint["before"]["complexity"]
            group = f"{case['alpha']} {checkpoint['phase']}"
            branches = checkpoint["outcomes"]
            endpoint = [complexity(branch["native_fourth"]) for branch in branches]
            for policy in checkpoint["policies"]:
                if policy["name"] == "sparse_lookahead_hard":
                    continue
                probabilities = policy["probabilities"]
                if endpoint[0] is not None and all(
                    p == 0 or value is not None for p, value in zip(probabilities, endpoint)
                ):
                    difference = sum(
                        p * (value - endpoint[0])
                        for p, value in zip(probabilities, endpoint) if p > 0
                    )
                    transfers[group][policy["name"]][case["seed"]].append(difference)
            for branch in branches:
                first = complexity(branch["sp"])
                second = complexity(branch["native_second"])
                fourth = complexity(branch["native_fourth"])
                if first is None or second is None:
                    by_group[group]["no_converged_second"] += 1
                    continue
                kind = "undo" if branch["immediate_undo"] else "other"
                by_group[group][kind] += 1
                errors[group, kind]["two_step_reversal_mae"].append(abs(second - before))
                if fourth is not None:
                    errors[group, kind]["four_step_reversal_mae"].append(abs(fourth - before))
    return {
        group: {
            "counts": counts,
            "four_steps_native_followup": {
                name: paired_summary(effects) for name, effects in transfers[group].items()
            },
            **{
                kind: {name: mean(values) for name, values in errors[group, kind].items()}
                for kind in ("undo", "other")
            },
        }
        for group, counts in sorted(by_group.items())
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot", type=Path)
    parser.add_argument("heldout", type=Path)
    parser.add_argument("--diagnostic", type=Path)
    args = parser.parse_args()
    pilot_cases, heldout_cases = map(read_cases, (args.pilot, args.heldout))
    assert not ({case["seed"] for case in pilot_cases} & {case["seed"] for case in heldout_cases})
    training, testing = map(examples, (pilot_cases, heldout_cases))
    result = {}
    for group, rows in sorted(testing.items()):
        xbar, ybar, slope = fit(training[group])
        result[f"{group[0]} {group[1]}"] = {
            "training_branches": len(training[group]),
            "test_branches": len(rows),
            "test_formulas": len({row[0] for row in rows}),
            "slope": slope,
            "intercept": ybar - slope * xbar,
            "heldout_mae": {
                "zero": mean_absolute_error(rows, lambda _: 0.0),
                "pilot_group_mean": mean_absolute_error(rows, lambda _: ybar),
                "reversal": mean_absolute_error(rows, lambda x: -x),
                "pilot_linear": mean_absolute_error(rows, lambda x: ybar + slope * (x - xbar)),
            },
        }
    if args.diagnostic:
        diagnostic_cases = read_cases(args.diagnostic)
        assert {(c["alpha"], c["seed"]) for c in diagnostic_cases} == {
            (c["alpha"], c["seed"]) for c in heldout_cases
        }
        result["reversal_diagnostic"] = diagnostic(diagnostic_cases)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
