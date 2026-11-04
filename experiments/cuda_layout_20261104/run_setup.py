"""Run standalone Rust layouts in fresh, alternated Linux processes."""
import json
import pathlib
import random
import subprocess

root = pathlib.Path(__file__).resolve().parent
subprocess.run(['rustc', '-O', '--edition=2021', str(root / 'setup.rs'), '-o', str(root / 'setup')], check=True)
randomizer = random.Random(30)
rows = []
for n in [50000, 100000, 1000000]:
    for trial in range(9):
        seed = [3, 7, 11][trial % 3]
        variants = ['sort', 'linear']
        randomizer.shuffle(variants)
        paired = []
        for variant in variants:
            result = subprocess.run([str(root / 'setup'), variant, str(n), str(seed)], capture_output=True, text=True, check=True)
            row = json.loads(result.stdout)
            row['trial'] = trial
            rows.append(row)
            paired.append(row)
        assert paired[0]['hash'] == paired[1]['hash'], paired
        (root / 'results' / 'setup.json').write_text(json.dumps(rows, indent=2) + '\n')
        print(n, trial, [(r['variant'], round(r['seconds'], 4), r['max_rss_kib']) for r in paired], flush=True)
