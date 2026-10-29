"""Solve DIMACS with the native Python package, or generate random 3-SAT."""
import argparse
from pathlib import Path
from pybsp_rust import Formula, solve

parser = argparse.ArgumentParser(__doc__)
parser.add_argument("formula", nargs="?", type=Path)
args = parser.parse_args()
formula = (Formula.from_dimacs(args.formula.read_text()) if args.formula else
           Formula.random(k=3, alpha=2.5, n=128, seed=3))
result = solve(formula=formula, seed=3, timeout_seconds=10)
print(result)
if result.assignment is None:
    raise SystemExit(f"No solution returned: {result.status}")
assert formula.is_satisfied(result.assignment)
print("Verified assignment:", result.assignment.tolist())
