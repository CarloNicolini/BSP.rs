"""Summarize independent process medians and paired speed ratios."""
import collections
import json
import pathlib
import statistics

root = pathlib.Path(__file__).resolve().parent
result = {}
rows = json.loads((root / 'results/setup.json').read_text())
for n in [50000, 100000, 1000000]:
    summary = {}
    for label in ['sort', 'linear']:
        data = [row for row in rows if row['n'] == n and row['variant'] == label]
        summary[label] = {key: statistics.median(r[key] for r in data) for key in ['seconds', 'setup_peak_bytes', 'peak_live_bytes', 'max_rss_kib']}
        summary[label]['trials'] = len(data)
    result.setdefault('setup', {})[n] = summary
for filename in ['native.json', 'pack4.json', 'observables.json']:
    path = root / 'results' / filename
    if not path.exists():
        continue
    rows = json.loads(path.read_text())
    for n in [50000, 100000, 1000000]:
        groups = collections.defaultdict(list)
        for row in rows:
            if row['n'] == n and 'gpu_seconds' in row:
                groups[(row['label'], row['round'], row['seed'])].append(row['gpu_seconds'] / (row['sweeps'] or row['passes']) * 1000)
        process_medians = {key: statistics.median(values) for key, values in groups.items()}
        summary = {}
        for label in sorted({key[0] for key in groups}):
            values = [value for key, value in process_medians.items() if key[0] == label]
            ratios = [process_medians[('baseline', rnd, seed)] / value for (name, rnd, seed), value in process_medians.items() if name == label and ('baseline', rnd, seed) in process_medians]
            summary[label] = {'median_ms': statistics.median(values), 'min_ms': min(values), 'max_ms': max(values), 'processes': len(values), 'paired_speedup_median': statistics.median(ratios) if ratios else None}
        result.setdefault(filename, {})[n] = summary
(root / 'results/summary.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
