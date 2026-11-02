"""Audit every saved noise run without integrating the plant again.

python audit_noise_experiments.py results/<batch> [--watch]
The optional watcher audits only fully archived runs from the manifest.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def close(actual, expected, name, atol=1e-11):
    require(np.allclose(actual, expected, rtol=1e-11, atol=atol), name)


def audit_run(batch, row, config):
    folder = batch / row['archive']
    trajectory = np.loadtxt(folder / 'trajectory.csv', delimiter=',', skiprows=1)
    controls = np.loadtxt(folder / 'control.csv', delimiter=',', skiprows=1)
    observations = np.loadtxt(folder / 'noise_control.csv', delimiter=',', skiprows=1)
    require(all(np.all(np.isfinite(a)) for a in (trajectory, controls, observations)), 'nonfinite raw data')
    require(trajectory.shape == (3001, 13), 'state record count / columns')
    require(controls.shape == (1500, 6) and observations.shape == (1500, 16), 'control record count / columns')
    close(trajectory[:, 0], np.arange(3001)*0.01, 'state timestamps')
    times = np.arange(1500)*0.02
    close(controls[:, 0], times, 'control timestamps')
    close(observations[:, 0], times, 'noise timestamps')
    close(observations[:, 1:5], controls[:, 1:5], 'true state vs control log')
    close(observations[:, 1:5], trajectory[::2, 2:6][:-1], 'true state vs trajectory')
    close(observations[:, 5:9], observations[:, 1:5]+observations[:, 9:13], 'measurement identity')
    close(observations[:, 15], observations[:, 13]+observations[:, 14], 'total force identity')
    close(controls[:, 5], observations[:, 15], 'saved control force semantics')
    require(np.max(np.abs(observations[:, 13])) <= 20+1e-10, 'actuator limit')
    initial = [0, 0, np.deg2rad(row['angle_deg']), 0]
    close(trajectory[0, 2:6], initial, 'initial true state')

    # Independently regenerate both independent streams from the saved protocol.
    input_seed, sensor_seed = np.random.SeedSequence(row['seed']).spawn(2)
    input_rng = np.random.default_rng(input_seed)
    unit_force = (input_rng.uniform(-np.sqrt(3), np.sqrt(3), 1500)
                  if config['input_distribution'] == 'uniform' else input_rng.standard_normal(1500))
    sensor_rng = np.random.default_rng(sensor_seed)
    unit_sensor = sensor_rng.standard_normal((1500, 4))
    sigma = np.array(config['sensor_std'], dtype=float)
    sigma[2:] = np.deg2rad(sigma[2:])
    force = (unit_force*config['input_std_N']*row['level']
             if row['kind'] == 'input' else np.zeros(1500))
    noise = (unit_sensor*sigma*row['level']
             if row['kind'] == 'sensor' else np.zeros((1500, 4)))
    force[times < config['noise_onset_s']] = 0
    noise[times < config['noise_onset_s']] = 0
    close(observations[:, 14], force, 'regenerated disturbance', 1e-15)
    close(observations[:, 9:13], noise, 'regenerated sensor noise', 1e-15)
    digest = hashlib.sha256(np.column_stack((times, force, noise)).tobytes()).hexdigest()
    require(digest == row['noise_schedule_sha256'], 'noise hash')
    if row['kind'] == 'input':
        require(observations[0, 14] != 0, 'startup input injection')
    elif row['kind'] == 'sensor':
        require(np.any(observations[0, 9:13] != 0), 'startup sensor injection')

    metadata = read_json(folder / 'metadata.json')
    require(metadata['noise_schedule_sha256'] == digest, 'metadata noise hash')
    archived_config = read_json(folder / 'run_config.json')
    require(archived_config['experiment'] == config, 'per-run protocol')
    require(archived_config['baseline_config'] == read_json(batch / 'baseline_configs' /
            f"{row['method']}_{row['angle_deg']}deg.json"), 'baseline config')

    tail = trajectory[:, 0] >= config['duration_s']-config['tail_seconds']-1e-10
    dt = np.diff(np.r_[times, config['duration_s']])
    tail_dt = np.maximum(0, np.r_[times[1:], config['duration_s']] -
                         np.maximum(times, config['duration_s']-config['tail_seconds']))
    computed = {
        'tail_angle_rms_deg': np.sqrt(np.mean(trajectory[tail, 6]**2)),
        'tail_position_rms_m': np.sqrt(np.mean(trajectory[tail, 2]**2)),
        'tail_angle_abs_p95_deg': np.quantile(np.abs(trajectory[tail, 6]), 0.95),
        'tail_actuator_rms_N': np.sqrt(np.sum(observations[:, 13]**2*tail_dt)/config['tail_seconds']),
        'integral_u2_N2_s': np.sum(observations[:, 13]**2*dt),
        'max_energy_balance_error_J': np.max(np.abs(trajectory[:, 12])),
    }
    for field, value in computed.items():
        close(value, row[field], 'recomputed metric '+field)
        close(value, read_json(folder / 'metrics.json')[field], 'archived metric '+field)
    require(row['upright_region_held'] == bool(np.all(np.abs(trajectory[:, 6]) < config['upright_region_limit_deg'])),
            'upright classification')
    files = {name: hashlib.sha256((folder/name).read_bytes()).hexdigest()
             for name in ('trajectory.csv', 'control.csv', 'noise_control.csv')}
    return {'archive': row['archive'], 'raw_csv_sha256': files,
            'max_energy_balance_error_J': computed['max_energy_balance_error_J']}


def audit(batch, watch=False):
    config = read_json(batch / 'config.json')
    checked, errors = {}, []
    while True:
        try:
            manifest = read_json(batch / 'manifest.json')
        except json.JSONDecodeError:
            if watch:
                time.sleep(0.2)
                continue
            raise
        for row in manifest['runs']:
            archive = row['archive']
            if archive in checked:
                continue
            try:
                require(row['run_status'] == 'completed', 'run failed')
                checked[archive] = audit_run(batch, row, config)
            except Exception as error:
                errors.append({'archive': archive, 'error': str(error)})
                checked[archive] = {'archive': archive, 'audit_failed': True}
            if len(checked) % 100 == 0:
                print(f'Audited {len(checked)} runs; issues={len(errors)}', flush=True)
        if manifest['status'] != 'running' or not watch:
            break
        time.sleep(0.5)
    expected = len(config['angles_deg'])*2*(1+2*len(config['levels'])*len(config['seeds']))
    if len(checked) != expected:
        errors.append({'error': f'expected {expected} runs, found {len(checked)}'})
    source_hashes = {}
    for name, digest in manifest['source_sha256'].items():
        data = (batch / 'source' / name).read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        source_hashes[name] = actual
        if actual != digest:
            errors.append({'error': 'source SHA-256 mismatch: '+name})
        if name in config['base_source_blobs']:
            git_digest = hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
            if git_digest != config['base_source_blobs'][name]:
                errors.append({'error': 'main Git blob mismatch: '+name})
    groups = {}
    for row in manifest['runs']:
        key = (row['angle_deg'], row['kind'], row['level'], row['seed'])
        groups.setdefault(key, {})[row['method']] = row
    pairs = 0
    for key, group in groups.items():
        if set(group) != {'pid', 'lqr'} or group['pid'].get('noise_schedule_sha256') != group['lqr'].get('noise_schedule_sha256'):
            errors.append({'error': 'paired-noise mismatch', 'condition': key})
        else:
            pairs += 1
    report = {'status': 'passed' if not errors else 'failed', 'expected_runs': expected,
              'audited_runs': len(checked), 'verified_pairs_including_baselines': pairs,
              'source_sha256': source_hashes, 'errors': errors,
              'max_energy_balance_error_J': max((r.get('max_energy_balance_error_J', 0) for r in checked.values()), default=0),
              'checks': ['all raw CSV timestamps, finite values and record counts', 'initial true states',
                         'true state, observation and total-force identities', '20 N actuator bound',
                         'independent regeneration of every noise sequence, including startup injection',
                         'matching PID/LQR schedule hashes', 'per-run protocols and baseline configurations',
                         'recomputed RMS, p95 angle, control effort and energy residual',
                         'upright classification', 'archived source SHA-256 and original core Git blob hashes'],
              'runs': list(checked.values())}
    (batch / 'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'runs'}, ensure_ascii=False, indent=2), flush=True)
    return not errors


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch', type=Path)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    raise SystemExit(0 if audit(args.batch.resolve(), args.watch) else 1)
