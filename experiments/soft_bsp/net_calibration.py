"""Test whether first-point scores can predict relative net-depth continuation."""
import argparse
import json
from pathlib import Path
from statistics import mean

from analyze import complexity, paired_summary
from net_analysis import read_runs


def complete_checkpoints(runs, phase):
    for (seed, arm), run in runs.items():
        if arm != "native":
            continue
        for point in run["checkpoints"]:
            if point["phase"] != phase:
                continue
            branches = point["branches"]
            child = [complexity(branch["first"]) for branch in branches]
            endpoint = [
                branch["continuation"]["complexity"] if branch["continuation"] else None
                for branch in branches
            ]
            if None in child or None in endpoint:
                continue
            prior = [branch["action"]["prior"] for branch in branches]
            # Differencing by the native candidate removes the common
            # baseline level and preserves the quantity used in ranking.
            yield seed, [
                (b - prior[0], f - child[0], v - endpoint[0])
                for b, f, v in zip(prior, child, endpoint)
            ]


def fit(checkpoints):
    # Two coefficients and no intercept. Equal weight per candidate.
    # Ridge strength is 5% of the average diagonal of X'X.
    rows = [row for _, rows in checkpoints for row in rows[1:]]
    prior2 = sum(b * b for b, _, _ in rows)
    delta2 = sum(f * f for _, f, _ in rows)
    cross = sum(b * f for b, f, _ in rows)
    regularizer = 0.025 * (prior2 + delta2)
    a, d = prior2 + regularizer, delta2 + regularizer
    rhs_b = sum(b * v for b, _, v in rows)
    rhs_f = sum(f * v for _, f, v in rows)
    determinant = a * d - cross * cross
    return (
        (d * rhs_b - cross * rhs_f) / determinant,
        (a * rhs_f - cross * rhs_b) / determinant,
    ) if determinant > 0 else (0.0, 0.0)


def evaluate(test, coef):
    approaches = {
        "zero_relative": lambda b, f: 0.0,
        "immediate": lambda b, f: f,
        "calibrated": lambda b, f: coef[0] * b + coef[1] * f,
    }
    output = {}
    for name, predict in approaches.items():
        errors, effects, regrets = {}, {}, {}
        for seed, rows in test:
            predictions = [predict(b, f) for b, f, _ in rows]
            choice = max(range(len(rows)), key=lambda i: rows[i][0] + predictions[i])
            optimal = max(b + v for b, _, v in rows)
            errors.setdefault(seed, []).append(mean(
                abs(prediction - v) for prediction, (_, _, v) in zip(predictions[1:], rows[1:])
            ))
            effects.setdefault(seed, []).append(rows[choice][2])
            regrets.setdefault(seed, []).append(optimal - (rows[choice][0] + rows[choice][2]))
        output[name] = {
            "mae": paired_summary(errors),
            "net_complexity_vs_native": paired_summary(effects),
            "regret_vs_evaluated_best_score": paired_summary(regrets),
        }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("development", type=Path)
    parser.add_argument("heldout", type=Path)
    args = parser.parse_args()
    development = read_runs(args.development, "development")
    heldout = read_runs(args.heldout, "heldout")
    result = {}
    for n in (640, 1280):
        result[n] = {}
        for phase in ("fix", "release"):
            training = list(complete_checkpoints(development[n], phase))
            testing = list(complete_checkpoints(heldout[n], phase))
            coef = fit(training)
            result[n][phase] = {
                "train_checkpoints": len(training),
                "test_checkpoints": len(testing),
                "coefficients_prior_and_immediate": coef,
                "holdout": evaluate(testing, coef),
            }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
