"""Replay one run with the batch's archived Python sources and configuration.

python replay_noise_run.py results/<batch> runs/sensor/angle_5/level_1/seed_0/lqr --output results/<new-replay>
The output must be a new directory. CSV hashes are compared byte for byte.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def replay(batch, archive, output):
    manifest = json.loads((batch / 'manifest.json').read_text())
    for name, digest in manifest['source_sha256'].items():
        if hashlib.sha256((batch / 'source' / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Archived source hash mismatch: '+name)
    original = (batch / archive).resolve()
    if batch not in original.parents:
        raise ValueError('Run path must stay inside the batch.')
    saved = json.loads((original / 'run_config.json').read_text())
    sys.path.insert(0, str(batch / 'source'))
    spec = importlib.util.spec_from_file_location('noise_archived_runner', batch / 'source' / 'run_noise_experiments.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    task = {key: saved[key] for key in ('angle_deg', 'kind', 'level', 'seed', 'method', 'archive')}
    task.update(config=saved['experiment'], baseline=saved['baseline_config'], directory=str(output))
    row = runner.run_one(task)
    if row['run_status'] != 'completed':
        raise RuntimeError(row.get('error', 'Replay failed'))
    comparisons = {}
    for name in ('trajectory.csv', 'control.csv', 'noise_control.csv'):
        before = hashlib.sha256((original/name).read_bytes()).hexdigest()
        after = hashlib.sha256((output/name).read_bytes()).hexdigest()
        comparisons[name] = {'original_sha256': before, 'replay_sha256': after, 'byte_identical': before == after}
    report = {'status': 'passed' if all(v['byte_identical'] for v in comparisons.values()) else 'different',
              'archive': archive, 'comparisons': comparisons,
              'note': 'Byte identity was checked in the same software environment. Other library/platform versions may differ within integration tolerance.'}
    (output / 'replay_verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    return report['status'] == 'passed'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch', type=Path)
    parser.add_argument('archive')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    raise SystemExit(0 if replay(args.batch.resolve(), args.archive, args.output.resolve()) else 1)
