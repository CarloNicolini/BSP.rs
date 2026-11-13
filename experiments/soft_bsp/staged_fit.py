"""Fit and assess phase-specific logistic models using the Python standard library."""
import math
from collections import defaultdict
from statistics import mean, pstdev


def sigmoid(x):
    t = math.exp(-abs(x))
    return 1 / (1 + t) if x >= 0 else t / (1 + t)


def solve(matrix, rhs):
    rows = [list(row) + [b] for row, b in zip(matrix, rhs)]
    for i in range(len(rows)):
        j = max(range(i, len(rows)), key=lambda j: abs(rows[j][i]))
        rows[i], rows[j] = rows[j], rows[i]
        pivot = rows[i][i]
        rows[i] = [v / pivot for v in rows[i]]
        for j, row in enumerate(rows):
            if i != j:
                factor = row[i]
                rows[j] = [x - factor * y for x, y in zip(row, rows[i])]
    return [row[-1] for row in rows]


def fit(rows, count):
    xs, ys = zip(*rows)
    means = [mean(x[i] for x in xs) for i in range(count)]
    scales = [pstdev(x[i] for x in xs) or 1.0 for i in range(count)]
    xs = [[1.0, *((x[i] - means[i]) / scales[i] for i in range(count))] for x in xs]
    average = min(1 - 1e-9, max(1e-9, mean(ys)))
    coefficients = [math.log(average / (1 - average)), *([0.0] * count)]
    penalty = 0.01
    if min(ys) != max(ys) or ys[0] not in (0, 1):
        for _ in range(80):
            size = count + 1
            gradient = [0.0] * size
            hessian = [[0.0] * size for _ in range(size)]
            for x, y in zip(xs, ys):
                p = sigmoid(sum(w * v for w, v in zip(coefficients, x)))
                for i in range(size):
                    gradient[i] += (p - y) * x[i] / len(xs)
                    for j in range(size):
                        hessian[i][j] += p * (1 - p) * x[i] * x[j] / len(xs)
            for i in range(1, size):
                gradient[i] += penalty * coefficients[i]
                hessian[i][i] += penalty
            delta = solve(hessian, gradient)
            step = max(1.0, max(map(abs, delta)) / 2)
            coefficients = [w - d / step for w, d in zip(coefficients, delta)]
            if max(map(abs, delta)) < 1e-8:
                break
    return {"mean": means, "scale": scales, "intercept": coefficients[0], "weights": coefficients[1:]}


def target(branch):
    return mean(label["status"] == "sat" for label in branch["labels"])


def predict(model, features):
    return sigmoid(model["intercept"] + sum(
        w * (x - m) / s for w, x, m, s in zip(
            model["weights"], features, model["mean"], model["scale"]
        )
    ))


def train(records):
    rows = defaultdict(list)
    for record in records:
        for point in record["checkpoints"]:
            rows[point["phase"]].extend(
                (branch["features"], target(branch)) for branch in point["branches"]
                if branch["features"] is not None
            )
    if any(not rows[phase] for phase in ("fix", "release")):
        raise ValueError("Both phases need at least one converged training candidate")
    return {
        name: {phase: fit(rows[phase], count) for phase in ("fix", "release")}
        for name, count in (("bias_depth", 2), ("full", 6))
    }


def assess(records, models):
    groups = defaultdict(list)
    for record in records:
        for point in record["checkpoints"]:
            groups[record["job"]["n"], point["phase"]].append(point)
    output = {}
    for (n, phase), points in sorted(groups.items()):
        complete = [
            point for point in points if all(branch["features"] is not None for branch in point["branches"])
        ]
        report = {"checkpoints": len(points), "complete_features": len(complete)}
        offsets = (1, 3, 8)
        report["net_depth"] = {}
        for i, offset in enumerate(offsets):
            spreads = []
            for point in points:
                values = [
                    branch["trace"]["complexity"][i] if branch["trace"] else None
                    for branch in point["branches"]
                ]
                if None not in values:
                    spreads.append(max(values) - min(values))
            report["net_depth"][offset] = {
                "complete": len(spreads), "spread_below_1e-6": sum(v < 1e-6 for v in spreads),
            }
        report["immediate_undo"] = sum(
            bool(branch["trace"] and branch["trace"]["undo"])
            for point in points for branch in point["branches"]
        )
        report["label_ties"] = sum(
            len({target(branch) for branch in point["branches"]}) == 1 for point in complete
        )
        for name, model in models.items():
            errors, correct, gain = [], [], []
            for point in complete:
                probabilities = [predict(model[phase], b["features"]) for b in point["branches"]]
                empirical = [target(b) for b in point["branches"]]
                choice = max(range(len(empirical)), key=probabilities.__getitem__)
                errors.extend(
                    (p - int(label["status"] == "sat")) ** 2
                    for p, branch in zip(probabilities, point["branches"]) for label in branch["labels"]
                )
                if len(set(empirical)) > 1:
                    correct.append(empirical[choice] == max(empirical))
                gain.append(empirical[choice] - empirical[0])
            report[name] = {
                "brier": mean(errors) if errors else None,
                "best_move_rate_on_non_ties": mean(correct) if correct else None,
                "target_gain_vs_native": mean(gain) if gain else None,
            }
        output[f"{n}/{phase}"] = report
    return output
