"""Paired input-force and state-measurement noise experiments.

No simulation by default. --preview --run executes the endpoint pilot only.
The original nonlinear simulator and both controller implementations are reused.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import csv
from datetime import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np

from cartpole_model import Parameters
from cartpole_simulation import SolverSettings, simulate_feedback
from feedback import (create_pid_controller, create_lqr_controller,
                      tune_angle_pid, tune_position_pid)


ROOT = Path(__file__).resolve().parent
STATE_NAMES = ['position_m', 'velocity_m_s', 'angle_rad', 'angular_velocity_rad_s']
THRESHOLDS = np.array([0.01, 0.01, np.deg2rad(0.1), np.deg2rad(0.1)])
LOG_NAMES = (['time_s'] + ['true_' + n for n in STATE_NAMES]
             + ['measured_' + n for n in STATE_NAMES]
             + ['noise_' + n for n in STATE_NAMES]
             + ['actuator_force_N', 'disturbance_force_N', 'total_force_N'])


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def validate_config(config):
    c = deepcopy(config)
    for key in ('duration_s', 'tail_seconds', 'input_std_N'):
        if not np.isfinite(c[key]) or c[key] <= 0:
            raise ValueError(key + ' must be finite and positive.')
    if c['tail_seconds'] > c['duration_s']:
        raise ValueError('tail_seconds exceeds duration_s.')
    if not np.isfinite(c['noise_onset_s']) or not 0 <= c['noise_onset_s'] < c['duration_s']:
        raise ValueError('noise_onset_s must lie in [0, duration).')
    if c['input_distribution'] not in ('gaussian', 'uniform'):
        raise ValueError('input_distribution must be gaussian or uniform.')
    if c['sample_period_s'] != 0.02:
        raise ValueError('This review fixes the common sampling period at 0.02 s.')
    s = np.asarray(c['sensor_std'], dtype=float)
    if s.shape != (4,) or not np.all(np.isfinite(s)) or np.any(s < 0):
        raise ValueError('sensor_std must contain four finite nonnegative standard deviations.')
    if c.get('sensor_std_units') != ['m', 'm/s', 'deg', 'deg/s']:
        raise ValueError('sensor_std_units must be [m, m/s, deg, deg/s].')
    allowed = {5, 10, 15, 20, 25, 30}
    for key in ('angles_deg', 'preview_angles_deg'):
        values = c[key]
        if not values or len(set(values)) != len(values) or not set(values) <= allowed:
            raise ValueError(key + ' must be a nonempty unique subset of [5,10,15,20,25,30].')
    if not set(c['preview_angles_deg']) <= set(c['angles_deg']):
        raise ValueError('Preview angles must be a subset of the full design.')
    if c['noise_onset_s'] > c['duration_s'] - c['tail_seconds']:
        raise ValueError('The full tail metric window must follow noise onset.')
    if not np.isfinite(c['upright_region_limit_deg']) or not 30 < c['upright_region_limit_deg'] <= 180:
        raise ValueError('upright_region_limit_deg must lie in (30,180].')
    for key in ('seeds', 'preview_seeds'):
        values = c[key]
        if not values or len(set(values)) != len(values) or any(type(v) is not int or not 0 <= v < 2**32 for v in values):
            raise ValueError(key + ' must contain unique nonnegative integer seeds below 2**32.')
    for key in ('levels', 'preview_levels'):
        values = c[key]
        if not values or len(set(values)) != len(values) or any(not np.isfinite(v) or v <= 0 for v in values):
            raise ValueError(key + ' must contain unique finite positive multipliers.')
    if not set(c['preview_seeds']) <= set(c['seeds']) or not set(c['preview_levels']) <= set(c['levels']):
        raise ValueError('Preview seeds and levels must be subsets of the full design.')
    if c['plot_level'] not in c['levels']:
        raise ValueError('plot_level must be a configured noise multiplier.')
    if c['plot_level'] not in c['preview_levels']:
        raise ValueError('plot_level must also be present in preview_levels.')
    return c


def verify_sources(config, root=ROOT):
    """Refuse silently running different controllers under a pinned commit label."""
    for name, expected in config['base_source_blobs'].items():
        path = (root / name).resolve()
        if root.resolve() not in path.parents:
            raise ValueError('Pinned source path must be inside the project.')
        data = path.read_bytes()
        actual = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        if actual != expected:
            raise ValueError(f'Pinned baseline changed: {name}. Review it before updating the manifest.')


def baseline_configs(root=ROOT):
    """Discover matched angle configurations from the current main naming scheme."""
    result = {}
    for angle in (5, 10, 15, 20, 25, 30):
        candidates = []
        for path in sorted((root / 'configs').glob('pid_[0-9][0-9]_*.json')):
            candidate = json.loads(path.read_text(encoding='utf-8'))
            if candidate['initial_state'] == dict(position_m=0, velocity_m_s=0, angle_deg=angle, angular_velocity_deg_s=0):
                candidates.append(path)
        if len(candidates) != 1:
            raise ValueError(f'Expected one PID pure-angle configuration for {angle} degrees.')
        pid_path = candidates[0]
        pair = {}
        for method in ('pid', 'lqr'):
            path = pid_path if method == 'pid' else pid_path.with_name('lqr_' + pid_path.name[4:])
            pair[method] = json.loads(path.read_text(encoding='utf-8'))
            control = pair[method]['control']
            if control['sample_period_s'] != 0.02:
                raise ValueError('This draft fixes Ts at 0.02 s; figure labels use that period.')
            if pair[method]['parameters'] != {'M': 1.0, 'm': 1.0, 'L': 0.5, 'g': 9.81}:
                raise ValueError('This draft fixes the nominal physical model used in the existing baseline.')
            if control['factory'] != f'create_{method}_controller':
                raise ValueError('Unexpected controller factory.')
            if (path.parent / control['file']).resolve() != (root / 'feedback.py').resolve():
                raise ValueError('Baseline must use this project feedback.py.')
            if control['settings']['force_limit_N'] != 20:
                raise ValueError('This draft fixes the common actuator limit at 20 N.')
        for key in ('parameters', 'initial_state', 'simulation', 'output'):
            if pair['pid'][key] != pair['lqr'][key]:
                raise ValueError(f'Baseline {angle}: PID/LQR {key} differ.')
        if pair['pid']['control']['sample_period_s'] != pair['lqr']['control']['sample_period_s']:
            raise ValueError('PID/LQR sampling periods differ.')
        if pair['pid']['control']['settings']['position_target_m'] != 0:
            raise ValueError('The common target must be zero.')
        if pair['pid']['initial_state'] != dict(position_m=0, velocity_m_s=0, angle_deg=angle, angular_velocity_deg_s=0):
            raise ValueError('Unexpected baseline initial state.')
        result[angle] = pair
    for method in ('pid', 'lqr'):
        common = None
        for angle in result:
            current = deepcopy(result[angle][method]); current.pop('initial_state')
            if common is not None and current != common:
                raise ValueError('Baseline tuning/conditions differ between initial angles.')
            common = current
    return result


def sample_times(duration, period):
    tolerance = 8 * np.finfo(float).eps * duration
    times = np.arange(int(np.floor(duration / period)) + 1) * period
    return times[times < duration - tolerance]


def noise_schedule(config, period, kind, level, seed):
    """Independent streams; no controller or initial-angle identifier in seed."""
    if kind not in ('baseline', 'input', 'sensor'):
        raise ValueError('Unknown noise injection kind.')
    times = sample_times(config['duration_s'], period)
    input_seed, sensor_seed = np.random.SeedSequence(seed).spawn(2)
    input_rng, sensor_rng = np.random.default_rng(input_seed), np.random.default_rng(sensor_seed)
    if config['input_distribution'] == 'uniform':
        unit_force = input_rng.uniform(-np.sqrt(3), np.sqrt(3), len(times))
    else:
        unit_force = input_rng.standard_normal(len(times))
    force = unit_force * config['input_std_N'] * level if kind == 'input' else np.zeros(len(times))
    sigma = np.asarray(config['sensor_std'], dtype=float).copy()
    sigma[2:] = np.deg2rad(sigma[2:])
    unit_measurement = sensor_rng.standard_normal((len(times), 4))
    measurement = unit_measurement * sigma * level if kind == 'sensor' else np.zeros((len(times), 4))
    active = times >= config['noise_onset_s']
    force[~active] = 0; measurement[~active] = 0
    digest = hashlib.sha256(np.column_stack((times, force, measurement)).tobytes()).hexdigest()
    return times, force, measurement, digest


class NoisyFeedback:
    """Exact state -> noisy observation -> bounded actuator -> external force."""
    def __init__(self, controller, times, disturbance, measurement):
        self.controller, self.times = controller, times
        self.disturbance, self.measurement = disturbance, measurement
        self.index, self.rows = 0, []

    def __call__(self, t, state):
        k = self.index
        if k >= len(self.times) or not np.isclose(t, self.times[k], rtol=0, atol=1e-10):
            raise RuntimeError('Noise callback timing does not match the pre-generated schedule.')
        measured = state + self.measurement[k]
        actuator = float(self.controller(t, measured.copy()))
        if not np.isfinite(actuator) or abs(actuator) > 20 + 1e-10:
            raise ValueError('Controller output exceeded the common actuator limit.')
        external = float(self.disturbance[k]); total = actuator + external
        self.rows.append(np.r_[t, state, measured, self.measurement[k], actuator, external, total])
        self.index += 1
        return total  # The original simulator integrates total_force * velocity as work.


def noise_metrics(result, rows, config):
    rows = np.asarray(rows)
    times = result.time; tail = times >= times[-1] - config['tail_seconds'] - 1e-10
    truth = result.state
    in_band = np.all(np.abs(truth) < THRESHOLDS, axis=1)
    intervals = np.diff(np.r_[rows[:, 0], times[-1]])
    actuator = rows[:, -3]
    tail_left = times[-1] - config['tail_seconds']
    tail_intervals = np.maximum(0, np.r_[rows[1:, 0], times[-1]] - np.maximum(rows[:, 0], tail_left))
    stats = {'baseline_tolerance_pass': bool(np.all(in_band[tail])),
             'tail_band_occupancy': float(np.mean(in_band[tail])),
             'tail_angle_rms_deg': float(np.rad2deg(np.sqrt(np.mean(truth[tail, 2]**2)))),
             'tail_position_rms_m': float(np.sqrt(np.mean(truth[tail, 0]**2))),
             'tail_angle_abs_p95_deg': float(np.rad2deg(np.quantile(np.abs(truth[tail, 2]), 0.95))),
             'tail_actuator_rms_N': float(np.sqrt(np.sum(actuator**2 * tail_intervals) / config['tail_seconds'])),
             'integral_u2_N2_s': float(np.sum(actuator**2 * intervals)),
             'actuator_tv_rate_N_s': float(np.sum(np.abs(np.diff(actuator))) / times[-1]),
             'peak_actuator_force_N': float(np.max(np.abs(actuator))),
             'saturation_fraction': float(np.sum(intervals[np.abs(actuator) >= 20 - 1e-9]) / times[-1]),
             'max_abs_position_m': float(np.max(np.abs(truth[:, 0]))),
             'max_abs_angle_deg': float(np.rad2deg(np.max(np.abs(truth[:, 2])))),
             'max_energy_balance_error_J': float(np.max(np.abs(result.energy_balance_error))),
             'state_records': len(times), 'control_records': len(rows)}
    # This is a declared finite-horizon upright-region check, not a proof of stability.
    exited = np.flatnonzero(np.abs(truth[:, 2]) >= np.deg2rad(config['upright_region_limit_deg']))
    stats['upright_region_held'] = bool(len(exited) == 0)
    stats['upright_region_first_exit_s'] = float(times[exited[0]]) if len(exited) else None
    active = times >= config['noise_onset_s'] - 1e-10
    for column, name in [(0, 'position_m_s'), (2, 'angle_rad_s')]:
        values = np.abs(truth[:, column])
        stats['IAE_' + name] = float(np.sum((values[1:] + values[:-1]) * np.diff(times) / 2))
        ta, ya = times[active], values[active]
        stats['post_onset_IAE_' + name] = float(np.sum((ya[1:] + ya[:-1]) * np.diff(ta) / 2))
    return stats


def add_baseline_deltas(rows):
    """Do not confuse nominal recovery quality with degradation caused by noise."""
    clean = {(r['angle_deg'], r['method']): r for r in rows
             if r['kind'] == 'baseline' and r['run_status'] == 'completed'}
    fields = ('tail_angle_rms_deg', 'tail_position_rms_m', 'tail_actuator_rms_N',
              'IAE_angle_rad_s', 'IAE_position_m_s',
              'post_onset_IAE_angle_rad_s', 'post_onset_IAE_position_m_s')
    for row in rows:
        base = clean.get((row['angle_deg'], row['method']))
        for field in fields:
            row['delta_' + field] = (row[field] - base[field]
                                     if base is not None and row['run_status'] == 'completed'
                                     and field in row and field in base else None)
    return rows


def controller_design(baseline):
    """Archive the actual fixed design, without running any plant dynamics."""
    p = baseline['pid']['parameters']; params = Parameters(p['M'], p['m'], p['L'], p['g'])
    period = baseline['pid']['control']['sample_period_s']
    pid = baseline['pid']['control']['settings']
    angle = tune_angle_pid(params, pid)
    position = tune_position_pid(params, pid, angle, period)
    settings = deepcopy(baseline['lqr']['control']['settings']); settings.pop('force_limit_N', None)
    raw = create_lqr_controller(params, settings, period)
    gain = [-float(raw(0, state)) for state in np.eye(4)]
    return {'parameters': p, 'sample_period_s': period,
            'pid': {'tuning_poles': pid['tuning_poles'], 'angle_pid': angle, 'position_pid': position},
            'lqr': {**settings, 'K': [gain]}, 'force_limit_N': 20}


def run_one(task):
    config, baseline = task['config'], task['baseline']
    out = Path(task['directory']); out.mkdir(parents=True, exist_ok=False)
    label = {k: task[k] for k in ('angle_deg', 'kind', 'level', 'seed', 'method', 'archive')}
    write_json(out / 'run_config.json', {**label, 'experiment': config, 'baseline_config': baseline})
    try:
        p = baseline['parameters']; params = Parameters(p['M'], p['m'], p['L'], p['g'])
        period = baseline['control']['sample_period_s']
        factory = create_pid_controller if task['method'] == 'pid' else create_lqr_controller
        controller = factory(params, deepcopy(baseline['control']['settings']), period)
        times, external, sensor, digest = noise_schedule(config, period, task['kind'], task['level'], task['seed'])
        noisy = NoisyFeedback(controller, times, external, sensor)
        s = baseline['simulation']
        result = simulate_feedback(controller=noisy, params=params, sample_period=period,
                                   initial_state=[0, 0, np.deg2rad(task['angle_deg']), 0],
                                   duration=config['duration_s'], output_dt=s['output_dt_s'],
                                   solver=SolverSettings(s['max_step_s'], s['rtol'], s['atol']))
        assert noisy.index == len(times)
        rows = np.asarray(noisy.rows)
        result.metadata.update({'draft': bool(config.get('draft', True)), 'run': {'status': 'completed'}, 'noise_schedule_sha256': digest,
                                'noise_experiment': label,
                                'force_semantics': {'trajectory.csv.force_N': 'total actuator plus external force',
                                                    'control.csv.force_N': 'total actuator plus external force',
                                                    'noise_control.csv.actuator_force_N': 'bounded controller output; used for effort and saturation'},
                                'observation_semantics': 'trajectory/control states are true; controller observes true+measurement_noise'})
        result.metadata['control']['observation'] = 'Noisy four-state measurement; see noise_control.csv and run_config.json.'
        result.save(out)
        np.savetxt(out / 'noise_control.csv', rows, delimiter=',', header=','.join(LOG_NAMES), comments='', fmt='%.17g')
        metrics = noise_metrics(result, rows, config)
        write_json(out / 'metrics.json', metrics)
        return {**label, 'run_status': 'completed', 'noise_schedule_sha256': digest, **metrics}
    except Exception as error:
        write_json(out / 'metadata.json', {'run': {'status': 'failed', 'error': str(error)}})
        return {**label, 'run_status': 'failed', 'error': str(error),
                'baseline_tolerance_pass': False, 'upright_region_held': False}


def run_batch(config, output, workers=1):
    verify_sources(config)
    bases = baseline_configs()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'config.json', config)
    source = output / 'source'; source.mkdir()
    source_hashes = {}
    for name in ('cartpole_model.py', 'cartpole_simulation.py', 'feedback.py', 'run_noise_experiments.py', 'noise_review_plots.py'):
        if not (ROOT / name).exists():
            continue
        data = (ROOT / name).read_bytes(); (source / name).write_bytes(data)
        source_hashes[name] = hashlib.sha256(data).hexdigest()
    inputs = output / 'baseline_configs'; inputs.mkdir()
    for angle in config['angles_deg']:
        for method in ('pid', 'lqr'):
            write_json(inputs / f'{method}_{angle}deg.json', bases[angle][method])
    write_json(output / 'controller_design.json', controller_design(bases[config['angles_deg'][0]]))
    manifest = {'status': 'running', 'draft': bool(config.get('draft', True)), 'preview': config.get('preview', False),
                'base_commit': config['base_commit'], 'source_sha256': source_hashes,
                'versions': {'python': sys.version.split()[0], **{n: importlib.metadata.version(n) for n in ('numpy', 'scipy', 'matplotlib')}},
                'statistics': 'Tail metrics use true state; shaded response bands are sample SD, not confidence intervals.',
                'upright_region_limit_deg': config['upright_region_limit_deg'],
                'interpretation': 'Numerical completion, strict no-noise tolerance, and upright-region retention are distinct.',
                'runs': []}
    write_json(output / 'manifest.json', manifest)
    tasks = []
    for angle in config['angles_deg']:
        conditions = [('baseline', 0.0, config['seeds'][0])]
        conditions += [(kind, level, seed) for kind in ('input', 'sensor') for level in config['levels'] for seed in config['seeds']]
        for kind, level, seed in conditions:
            for method in ('pid', 'lqr'):
                archive = f'runs/{kind}/angle_{angle}/level_{level:g}/seed_{seed}/{method}'
                tasks.append(dict(config=config, baseline=bases[angle][method], angle_deg=angle, kind=kind,
                                  level=level, seed=seed, method=method, archive=archive, directory=str(output / archive)))
    def collect(row):
        manifest['runs'].append(row)
        write_json(output / 'manifest.json', manifest)
        print(f"[{len(manifest['runs'])}/{len(tasks)}] {row['kind']} {row['angle_deg']}deg "
              f"level={row['level']:g} seed={row['seed']} {row['method']} {row['run_status']}", flush=True)
    if workers == 1:
        for task in tasks:
            collect(run_one(task))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for future in as_completed([pool.submit(run_one, task) for task in tasks]):
                collect(future.result())
    rows = sorted(manifest['runs'], key=lambda r: (r['angle_deg'], r['kind'], r['level'], r['seed'], r['method']))
    add_baseline_deltas(rows)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (output / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n'); writer.writeheader(); writer.writerows(rows)
    manifest['status'] = 'completed' if all(r['run_status'] == 'completed' for r in rows) else 'completed_with_failed_runs'
    write_json(output / 'manifest.json', manifest)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/noise_experiments.json')
    parser.add_argument('--run', action='store_true', help='execute simulations; default prints the plan only')
    parser.add_argument('--preview', action='store_true', help='endpoint angles, three seeds, two levels; pilot only')
    parser.add_argument('--workers', type=int, choices=range(1, 5), default=1)
    parser.add_argument('--plots', choices=['AC', 'all', 'A', 'B', 'C', 'none'], default='AC')
    parser.add_argument('--plot-only', type=Path, help='render an existing batch without rerunning its simulations')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.plot_only:
        output = args.plot_only.resolve()
    else:
        try:
            config = validate_config(json.loads(args.config.read_text(encoding='utf-8')))
            bases = baseline_configs()
            verify_sources(config)
        except (OSError, ValueError, KeyError) as error:
            parser.error(str(error))
        config['preview'] = bool(args.preview)
        if args.preview:
            config['angles_deg'] = config['preview_angles_deg']
            config['seeds'], config['levels'] = config['preview_seeds'], config['preview_levels']
        count = len(config['angles_deg']) * 2 * (1 + 2 * len(config['levels']) * len(config['seeds']))
        print(json.dumps({'draft': bool(config.get('draft', True)), 'run': args.run, 'preview': args.preview, 'runs': count,
                          'angles_deg': config['angles_deg'], 'seeds': config['seeds'], 'levels': config['levels'],
                          'duration_s': config['duration_s'], 'input_std_N': config['input_std_N'],
                          'sensor_std': config['sensor_std'], 'force_limit_N': 20,
                          'noise_onset_s': config['noise_onset_s'], 'base_commit': config['base_commit'],
                          'input_distribution': config['input_distribution'],
                          'pid_tuning_poles': bases[5]['pid']['control']['settings']['tuning_poles'],
                          'Q_diag': bases[5]['lqr']['control']['settings']['Q_diag'],
                          'R': bases[5]['lqr']['control']['settings']['R']}, indent=2), flush=True)
        if not args.run:
            return 0
        stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S-%f')
        output = (args.output or ROOT / 'results' / ('noise-experiments-' + stamp)).resolve()
        run_batch(config, output, args.workers)
    if args.plots != 'none':
        from noise_review_plots import render_all
        styles = ('A', 'B', 'C') if args.plots == 'all' else ('A', 'C') if args.plots == 'AC' else (args.plots,)
        render_all(output, styles)
    print('Experiment output:', output, flush=True)
    return 0 if json.loads((output / 'manifest.json').read_text())['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
