"""Run matched PID/LQR initial-state experiments on the nonlinear plant.

python run_comparison.py                    # all eight pairs, data and plots
python run_comparison.py --animation mp4    # also export 20 fps animations
"""

import argparse
from copy import deepcopy
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
from scipy.signal import cont2discrete

from cartpole_model import Parameters, linear_state_space
from feedback import create_lqr_controller, tune_angle_pid, tune_position_pid
import run_simulation as runner


THRESHOLDS = {'position_m': 0.01, 'velocity_m_s': 0.01,
              'angle_rad': float(np.deg2rad(0.1)),
              'angular_velocity_rad_s': float(np.deg2rad(0.1))}
TAIL_SECONDS = 5.0


def load_pairs(root, cases):
    """Reject unmatched conditions before creating any experiment output."""
    pairs = []
    common = {}
    for case in cases:
        paths = sorted((root / 'configs').glob(f'pid_{case}_*.json'))
        if len(paths) != 1:
            raise ValueError(f'Case {case}: expected one PID configuration.')
        pid_path = paths[0]
        lqr_path = pid_path.with_name('lqr_' + pid_path.name[4:])
        configs = {method: json.loads(path.read_text(encoding='utf-8'))
                   for method, path in [('pid', pid_path), ('lqr', lqr_path)]}
        pid, lqr = configs['pid'], configs['lqr']
        for key in ('parameters', 'initial_state', 'simulation', 'output'):
            if pid[key] != lqr[key]:
                raise ValueError(f'Case {case}: PID/LQR {key} differ.')
        if pid['simulation']['duration_s'] < TAIL_SECONDS:
            raise ValueError('Duration must be at least five seconds.')
        controls = [configs[m]['control'] for m in ('pid', 'lqr')]
        if controls[0]['sample_period_s'] != controls[1]['sample_period_s']:
            raise ValueError(f'Case {case}: PID/LQR sample periods differ.')
        if controls[0]['settings']['force_limit_N'] != controls[1]['settings']['force_limit_N']:
            raise ValueError(f'Case {case}: PID/LQR force limits differ.')
        limit = float(controls[0]['settings']['force_limit_N'])
        if not np.isfinite(limit) or limit <= 0:
            raise ValueError('The common force limit must be finite and positive.')
        if controls[0]['settings']['position_target_m'] != 0:
            raise ValueError('This comparison requires the common zero target.')
        for method, path in [('pid', pid_path), ('lqr', lqr_path)]:
            control = configs[method]['control']
            if control['factory'] != f'create_{method}_controller':
                raise ValueError(f'Case {case}: unexpected {method} factory.')
            if (path.parent / control['file']).resolve() != (root / 'feedback.py').resolve():
                raise ValueError('Comparison configurations must use the current feedback.py.')
            other = deepcopy(configs[method]); other.pop('initial_state')
            if method in common and other != common[method]:
                raise ValueError(f'Case {case}: {method} tuning/conditions changed between cases.')
            common[method] = other
        pairs.append((case, {'pid': pid_path, 'lqr': lqr_path}, configs))
    return pairs


def summarize(directory, force_limit):
    """Use the saved state grid and exact ZOH control interval durations."""
    metadata = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))
    row = {'run_status': metadata['run']['status'], 'control_success': False}
    if row['run_status'] != 'completed':
        row['error'] = metadata['run'].get('error', '')
        return row
    state = np.genfromtxt(directory / 'trajectory.csv', delimiter=',', names=True)
    control = np.atleast_1d(np.genfromtxt(directory / 'control.csv', delimiter=',', names=True))
    t = state['time_s']
    tail = t >= t[-1] - TAIL_SECONDS - 1e-10
    good = np.ones(len(t), dtype=bool)
    for name, threshold in THRESHOLDS.items():
        good &= np.abs(state[name]) < threshold
        row['final_' + name] = float(state[name][-1])
        row['tail_max_abs_' + name] = float(np.max(np.abs(state[name][tail])))
    bad = np.flatnonzero(~good)
    entry = int(bad[-1] + 1) if len(bad) else 0
    settled = entry < len(t) and t[-1] - t[entry] >= TAIL_SECONDS - 1e-10
    row['control_success'] = bool(np.all(good[tail]))
    row['settling_time_s'] = float(t[entry]) if settled else ''
    for name, label in [('position_m', 'p_m_s'), ('angle_rad', 'theta_rad_s')]:
        y = np.abs(state[name])
        row['IAE_' + label] = float(np.sum((y[:-1] + y[1:]) * np.diff(t) / 2))
    row['max_abs_position_m'] = float(np.max(np.abs(state['position_m'])))
    row['max_abs_angle_deg'] = float(np.max(np.abs(state['angle_deg'])))
    durations = np.diff(np.r_[control['time_s'], t[-1]])
    u = control['force_N']
    row['peak_force_N'] = float(np.max(np.abs(u)))
    row['integral_u2_N2_s'] = float(np.sum(u * u * durations))
    row['saturation_fraction'] = float(np.sum(durations[np.abs(u) >= force_limit - 1e-9]) / (t[-1] - t[0]))
    row['max_abs_energy_balance_error_J'] = float(np.max(np.abs(state['energy_balance_error_J'])))
    row['state_records'] = len(t)
    row['control_records'] = len(control)
    return row


def save_design(configs, destination):
    p = configs['pid']['parameters']
    params = Parameters(p['M'], p['m'], p['L'], p['g'])
    Ts = configs['pid']['control']['sample_period_s']
    settings = deepcopy(configs['lqr']['control']['settings'])
    settings.pop('force_limit_N')
    raw = create_lqr_controller(params, settings, Ts)
    K = np.array([-raw(0, x) for x in np.eye(4)]).reshape(1, 4)
    Ad, Bd, *_ = cont2discrete(linear_state_space(params), Ts, method='zoh')
    poles = np.linalg.eigvals(Ad - Bd @ K)
    pid = configs['pid']['control']['settings']
    angle = tune_angle_pid(params, pid)
    position = tune_position_pid(params, pid, angle, Ts)
    design = {'parameters': p, 'sample_period_s': Ts,
              'force_limit_N': configs['lqr']['control']['settings']['force_limit_N'],
              'lqr': {**settings, 'K': K.tolist(),
                      'linear_discrete_poles': [[float(z.real), float(z.imag)] for z in poles],
                      'max_pole_magnitude': float(np.max(np.abs(poles)))},
              'pid': {'tuning_poles': pid['tuning_poles'], 'position_pid': position, 'angle_pid': angle},
              'note': 'Poles describe the unsaturated local linear model, not global nonlinear stability.'}
    destination.write_text(json.dumps(design, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def plot_pair(case, runs, rows, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), constrained_layout=True)
    fields = [('position_m', 'Position [m]'), ('velocity_m_s', 'Velocity [m/s]'),
              ('angle_deg', 'Angle [deg, unwrapped]'), ('angular_velocity_deg_s', 'Angular velocity [deg/s]')]
    for method, color in [('pid', '#176b8c'), ('lqr', '#b45729')]:
        if rows[method]['run_status'] != 'completed':
            continue
        states = np.genfromtxt(runs[method] / 'trajectory.csv', delimiter=',', names=True)
        control = np.atleast_1d(np.genfromtxt(runs[method] / 'control.csv', delimiter=',', names=True))
        for ax, (field, label) in zip(axes.flat, fields):
            ax.plot(states['time_s'], states[field], label=method.upper(), color=color)
            ax.set_ylabel(label)
        axes[2, 0].step(np.r_[control['time_s'], states['time_s'][-1]],
                        np.r_[control['force_N'], control['force_N'][-1]], where='post', label=method.upper(), color=color)
    axes[2, 0].set_ylabel('Applied force [N]')
    axes[2, 1].axis('off')
    text = ['Success: all four states within thresholds for the final 5 s.']
    for method in ('pid', 'lqr'):
        r = rows[method]
        settling = r.get('settling_time_s', '')
        text.append(f"{method.upper()}: {'PASS' if r['control_success'] else 'FAIL'}; "
                    f"settling time: {settling if settling != '' else 'not reached'} s")
    axes[2, 1].text(0, 0.9, '\n\n'.join(text), va='top', fontsize=10, wrap=True)
    for ax in list(axes.flat)[:5]:
        ax.grid(alpha=0.25); ax.set_xlabel('Time [s]'); ax.legend()
    fig.suptitle('Matched nonlinear PID / LQR comparison | case ' + case)
    fig.savefig(path, dpi=150); plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', nargs='+', choices=[f'{i:02d}' for i in range(1, 9)],
                        default=[f'{i:02d}' for i in range(1, 9)])
    parser.add_argument('--animation', choices=['none', 'mp4', 'gif'], default='none')
    args = parser.parse_args(argv)
    if len(set(args.cases)) != len(args.cases):
        parser.error('Do not repeat case numbers.')
    root = Path(__file__).resolve().parent
    try:
        pairs = load_pairs(root, args.cases)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    started = datetime.now(runner.TIMEZONE)
    batch = root / 'results' / ('comparison-' + started.strftime('%Y%m%d-%H%M%S-%f'))
    batch.mkdir(parents=True)
    inputs = batch / 'inputs'; inputs.mkdir()
    source = batch / 'source'; source.mkdir()
    hashes = {}
    for name in ('feedback.py', 'cartpole_model.py', 'cartpole_simulation.py',
                 'cartpole_visualization.py', 'run_simulation.py', 'run_comparison.py'):
        data = (root / name).read_bytes(); (source / name).write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    manifest = {'started_at': started.isoformat(), 'cases': args.cases, 'animation': args.animation,
                'thresholds_SI': THRESHOLDS, 'tail_seconds': TAIL_SECONDS,
                'angle_policy': 'Unwrapped angle; target is exactly zero, not modulo 2*pi.',
                'source_sha256': hashes, 'runs': [], 'status': 'running'}
    manifest_path = batch / 'comparison.json'
    def write_manifest():
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    write_manifest()
    save_design(pairs[0][2], batch / 'design.json')
    rows = []
    previous_root = runner.RESULTS_ROOT
    try:
        runner.RESULTS_ROOT = batch / 'runs'
        with tempfile.TemporaryDirectory(prefix='cartpole-comparison-') as tmp:
            for case, paths, configs in pairs:
                pair_runs, pair_rows = {}, {}
                for method in ('pid', 'lqr'):
                    shutil.copyfile(paths[method], inputs / paths[method].name)
                    config = deepcopy(configs[method])
                    config['control']['file'] = str((paths[method].parent / config['control']['file']).resolve())
                    config['output']['animation']['format'] = args.animation
                    config_path = Path(tmp) / paths[method].name
                    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
                    before = set(runner.RESULTS_ROOT.iterdir()) if runner.RESULTS_ROOT.exists() else set()
                    print(f'Case {case}: {method.upper()}', flush=True)
                    try:
                        runner.main(['--config', str(config_path)])
                    except SystemExit as error:
                        if error.code not in (0, None):
                            print(f'Case {case}: {method.upper()} run failed ({error.code}).', flush=True)
                    created = set(runner.RESULTS_ROOT.iterdir()) - before
                    if len(created) != 1:
                        raise RuntimeError('Expected exactly one new archive for this run.')
                    directory = created.pop()
                    summary = summarize(directory, config['control']['settings']['force_limit_N'])
                    row = {'case': case, 'method': method, 'archive': str(directory.relative_to(batch)), **summary}
                    rows.append(row); pair_runs[method] = directory; pair_rows[method] = row
                    manifest['runs'].append(row); write_manifest()
                plot_pair(case, pair_runs, pair_rows, batch / f'comparison_{case}.png')
        fields = list(dict.fromkeys(k for row in rows for k in row))
        with (batch / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
        manifest['status'] = 'completed' if all(r['run_status'] == 'completed' for r in rows) else 'completed_with_failed_runs'
        manifest['finished_at'] = datetime.now(runner.TIMEZONE).isoformat(); write_manifest()
    except Exception as error:
        manifest.update(status='failed', error=str(error)); write_manifest()
        raise
    finally:
        runner.RESULTS_ROOT = previous_root
    print('Comparison results:', batch.resolve(), flush=True)
    return 0 if manifest['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
