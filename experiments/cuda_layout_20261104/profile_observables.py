"""Nsight profile of the baseline observable pass, separate from timing trials."""
import pathlib
import subprocess
root = pathlib.Path(__file__).resolve().parent
for n in [50000, 100000, 1000000]:
    command = ['ncu', '--section', 'SpeedOfLight', '--section', 'MemoryWorkloadAnalysis', '--section', 'Occupancy', '--section', 'LaunchStats', '--section', 'SchedulerStats', '--kernel-name', 'regex:clause_observables|observable_final|variables<', '--launch-count', '3', '--force-overwrite', '-o', str(root / 'results' / f'observables_{n}'), str(root / 'baseline-bench'), str(n), '3', '0', '2']
    with (root / 'results' / f'ncu_observables_{n}.txt').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    report = root / 'results' / f'observables_{n}.ncu-repz'
    for page, extension in [('details', 'txt')]:
        result = subprocess.run(['ncu', '--import', str(report), '--page', page], capture_output=True, text=True, check=True)
        (root / 'results' / f'ncu_observable_details_{n}.{extension}').write_text(result.stdout)
    print(n, 'profiled', flush=True)
