"""Profile production kernels; replay measurements are excluded from timing trials."""
import pathlib
import subprocess
root = pathlib.Path(__file__).resolve().parent
binary = root.parents[1] / 'target/release/bsp-rust'
for n in [50000, 100000, 1000000]:
    command = ['ncu', '--target-processes', 'all', '--section', 'SpeedOfLight', '--section', 'MemoryWorkloadAnalysis', '--section', 'Occupancy', '--section', 'LaunchStats', '--section', 'WarpStateStats', '--section', 'SchedulerStats', '--launch-count', '8', '--force-overwrite', '-o', str(root / 'results' / f'baseline_{n}'), str(binary), '--generate', '3', '4.2', str(n), '--seed', '3', '--backend', 'cuda-f32', '--sweeps', '2', '--output', str(root / 'results' / 'profile_run.json'), '--no-progress']
    with (root / 'results' / f'ncu_baseline_{n}.txt').open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    print(n, result.returncode, flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
