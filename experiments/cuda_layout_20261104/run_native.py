"""Alternate native CUDA benchmark processes; each resets messages outside timing."""
import argparse
import json
import pathlib
import random
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('binaries', nargs='+', help='label=/absolute/path')
parser.add_argument('--rounds', type=int, default=3)
parser.add_argument('--mode', choices=['sweep', 'obs'], default='sweep')
parser.add_argument('--output', default='native.json')
a = parser.parse_args()
root = pathlib.Path(__file__).resolve().parent
variants = dict(value.split('=', 1) for value in a.binaries)
rows = []
rng = random.Random(30)
for n in [50000, 100000, 1000000]:
    for round_index in range(a.rounds):
        for seed in [3, 7, 11]:
            order = list(variants)
            rng.shuffle(order)
            expected = None
            for label in order:
                result = subprocess.run([variants[label], str(n), str(seed), '3', '50', a.mode], check=True, capture_output=True, text=True)
                data = [json.loads(line) for line in result.stdout.splitlines()]
                identity = data[-1]
                if expected is not None:
                    assert identity == expected, (label, expected, identity)
                expected = dict(identity)
                for row in data:
                    row.update(label=label, round=round_index)
                    rows.append(row)
                (root / 'results' / a.output).write_text(json.dumps(rows, indent=2)+'\n')
                print(n, round_index, seed, label, [round(r['gpu_seconds']/(r['sweeps'] or r['passes'])*1000, 4) for r in data[:-1]], identity['hash'], flush=True)
