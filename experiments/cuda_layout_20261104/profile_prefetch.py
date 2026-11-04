"""Compare ordered observable reductions under Nsight after unprofiled trials."""
import pathlib
import subprocess
root = pathlib.Path(__file__).resolve().parent
for label in ['obsbase', 'prefetch4', 'prefetch8']:
    report = root / 'results' / f'reduction_{label}'
    command = ['ncu', '--section', 'SpeedOfLight', '--section', 'MemoryWorkloadAnalysis', '--section', 'LaunchStats', '--section', 'SchedulerStats', '--section', 'WarpStateStats', '--kernel-name', 'regex:observable_final', '--launch-count', '1', '--force-overwrite', '-o', str(report), str(root / (label + '-bench')), '1000000', '3', '0', '2', 'obs']
    with (root / 'results' / f'ncu_reduction_{label}.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    result = subprocess.run(['ncu', '--import', str(report) + '.ncu-repz', '--page', 'details'], capture_output=True, text=True, check=True)
    (root / 'results' / f'ncu_reduction_{label}.txt').write_text(result.stdout)
    print(label, 'profiled', flush=True)
