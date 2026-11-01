"""JSON 驱动的 PID 对照实验；每次仿真沿用项目原有归档与非线性求解。"""

import argparse
from copy import deepcopy
from datetime import datetime
import importlib.metadata
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
from scipy.signal import cont2discrete

from cartpole_model import linear_state_space
from cartpole_simulation import simulate_feedback
import run_simulation as runner


def plot_summary(rows, study, folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    names = [v['name'] for v in study['variants']]
    cases = [c['name'] for c in study['cases']]
    duration = json.loads((folder / 'base_config.json').read_text())['simulation']['duration_s']
    tail = study['tail_duration_s']
    lookup = {(row['variant'], row['case']): row for row in rows}
    values = np.array([[2 if lookup[name, case]['success'] else
                        1 if lookup[name, case]['settling_time_s'] is not None else 0
                        for case in cases] for name in names])
    fig, ax = plt.subplots(figsize=(max(12, len(cases)*0.65), max(5.5, len(names)*0.44 + 3.2)))
    ax.imshow(values, aspect='auto', vmin=0, vmax=2,
              cmap=ListedColormap(['#d5d9df', '#efbd58', '#389e80']))
    labels = ['_'.join(case.split('_')[2:]) if case.startswith('pid_') else case for case in cases]
    ax.set_xticks(range(len(cases)), labels, rotation=55, ha='right')
    ax.set_yticks(range(len(names)), names)
    fig.suptitle(f'Nonlinear sampled PID | {duration:g} s trials, unchanged plant', y=0.98)
    fig.legend(handles=[Patch(color='#389e80', label=f'All last {tail:g} s within tolerance'),
                       Patch(color='#efbd58', label=f'Settled later; less than {tail:g} s observed'),
                       Patch(color='#d5d9df', label=f'Not settled by {duration:g} s')],
               loc='upper center', bbox_to_anchor=(0.5, 0.94), ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(folder / 'recovery_map.png', dpi=180)
    plt.close(fig)


def plot_comparison(rows, case, variants, folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for name in variants:
        row = next(r for r in rows if r['variant'] == name and r['case'] == case)
        output = runner.RESULTS_ROOT / row['directory']
        states = np.genfromtxt(output / 'trajectory.csv', delimiter=',', names=True)
        diag = np.genfromtxt(output / 'pid_diagnostics.csv', delimiter=',', names=True)
        axes[0, 0].plot(states['time_s'], states['position_m'], label=name)
        axes[0, 1].plot(states['time_s'], states['angle_deg'], label=name)
        axes[1, 0].step(diag['time_s'], diag['force_N'], where='post', label=name)
        axes[1, 1].plot(diag['time_s'], diag['position_i'], label=name)
    for ax, label in zip(axes.flat, ['Position [m]', 'Angle [deg]', 'Force [N]', 'Position I contribution [rad]']):
        ax.set_ylabel(label)
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel('Time [s]')
    fig.suptitle(f'{case} | Nonlinear sampled PID')
    fig.tight_layout()
    fig.savefig(folder / f'{case}_comparison.png', dpi=180)
    plt.close(fig)


def patch_settings(target, changes):
    for key, value in changes.items():
        if isinstance(value, dict):
            patch_settings(target[key], value)
        else:
            target[key] = value


def measure(result, settings, angle, position, tolerances, tail_duration, output):
    states = result.state.copy()
    states[:, 0] -= settings['position_target_m']
    states[:, 2:] = np.rad2deg(states[:, 2:])
    within = np.all(np.abs(states) < tolerances, axis=1)
    bad = np.flatnonzero(~within)
    settled = float(result.time[bad[-1] + 1]) if len(bad) and bad[-1] + 1 < len(within) else None
    if not len(bad):
        settled = 0.0
    success = bool(np.all(within[result.time >= result.time[-1] - tail_duration]))

    # 按控制记录重建 I 和限幅前指令；核对实际力，防止诊断与控制代码漂移。
    I = 0.0
    Ts = result.metadata['control']['sample_period_s']
    angle_limit = np.deg2rad(settings['angle_limit_deg'])
    force_limit = settings['force_limit_N']
    records = []
    for t, p, v, theta, omega, u in result.control_log:
        error = settings['position_target_m'] - p
        ref_raw = position['kp'] * error + I - position['kd'] * v
        ref = float(np.clip(ref_raw, -angle_limit, angle_limit))
        raw_force = angle['kp'] * (theta - ref) + angle['kd'] * omega
        np.testing.assert_allclose(np.clip(raw_force, -force_limit, force_limit), u, atol=1e-10)
        records.append([t, I, ref_raw, ref, raw_force, u])
        I = float(np.clip(I + position['ki'] * Ts * error,
                          -position['integral_limit'], position['integral_limit']))
    records = np.array(records)
    np.savetxt(output / 'pid_diagnostics.csv', records, delimiter=',', fmt='%.17g', comments='',
               header='time_s,position_i,theta_ref_raw,theta_ref,force_raw_N,force_N')
    weights = np.diff(np.r_[records[:, 0], result.time[-1]])
    return {
        'success': success, 'settling_time_s': settled,
        'max_position_m': float(np.max(np.abs(states[:, 0]))),
        'max_angle_deg': float(np.max(np.abs(states[:, 2]))),
        'force_rms_N': float(np.sqrt(np.average(records[:, 5]**2, weights=weights))),
        'force_saturation_fraction': float(np.average(np.abs(records[:, 4]) >= force_limit, weights=weights)),
        'reference_saturation_fraction': float(np.average(np.abs(records[:, 2]) >= angle_limit, weights=weights)),
        'integral_saturation_fraction': float(np.average(np.abs(records[:, 1]) >= position['integral_limit'] - 1e-12, weights=weights)),
        'max_energy_error_J': float(np.max(np.abs(result.energy_balance_error))),
        'final_state': states[-1].tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/archive/pid_tuning_study.json'))
    args = parser.parse_args()
    study = json.loads(args.config.read_text())
    base_path = (args.config.parent / study['base_config']).resolve()
    base = json.loads(base_path.read_text())
    # 固定本批次的反馈快照，避免实验中编辑源码造成混合版本。
    folder = runner.RESULTS_ROOT / datetime.now(runner.TIMEZONE).strftime('%Y%m%d-%H%M%S-%f')
    folder.mkdir(parents=True)
    (folder / 'study.json').write_bytes(args.config.read_bytes())
    (folder / 'base_config.json').write_bytes(base_path.read_bytes())
    (folder / 'analysis_source.py').write_bytes(Path(__file__).read_bytes())
    source = base_path.parent / base['control']['file']
    frozen = folder / 'feedback_source.py'
    frozen.write_bytes(source.read_bytes())
    namespace = {}
    exec(compile(frozen.read_bytes(), str(frozen), 'exec'), namespace)
    versions = {'python': sys.version.split()[0], **{
        key: importlib.metadata.version(key) for key in ['numpy', 'scipy']}}
    rows = []
    print(f'实验索引：{folder}', flush=True)
    with TemporaryDirectory(prefix='pid-study-') as tmp:
        path = Path(tmp) / 'config.json'
        for variant in study['variants']:
            for case in study['cases']:
                config = deepcopy(base)
                config['control']['file'] = str(frozen.resolve())
                patch_settings(config['control']['settings'], variant['settings'])
                config['initial_state'] = case['initial_state']
                config['output'] = {'plot': False, 'animation': {'format': 'none'}}
                path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n')
                output, manifest = runner._archive_run(path, datetime.now(runner.TIMEZONE))
                config, kwargs = runner.read_config(output / 'config.json')
                settings = config['control']['settings']
                angle = namespace['tune_angle_pid'](kwargs['params'], settings)
                position = namespace['tune_position_pid'](kwargs['params'], settings, angle, kwargs['sample_period'])
                Ad, Bd, _, _, _ = cont2discrete(linear_state_space(kwargs['params']), kwargs['sample_period'])
                ka, da = angle['kp'], angle['kd']
                kp, ki, kd = (position[key] for key in ['kp', 'ki', 'kd'])
                closed = np.block([[Ad + Bd @ np.array([[ka*kp, ka*kd, ka, da]]), -ka*Bd],
                                   [np.array([[-ki*kwargs['sample_period'], 0, 0, 0]]), np.ones((1, 1))]])
                result = simulate_feedback(**kwargs)
                result.metadata.update(manifest)
                result.metadata['versions'] = versions
                result.metadata['run'].update(status='completed', finished_at=datetime.now(runner.TIMEZONE).isoformat())
                result.save(output)
                metrics = measure(result, settings, angle, position, study['tolerances'], study['tail_duration_s'], output)
                row = {'variant': variant['name'], 'case': case['name'], 'directory': output.name,
                       'poles': settings['tuning_poles'], 'position_pid': position, 'angle_pid': angle,
                       'discrete_radius': float(max(abs(np.linalg.eigvals(closed)))), **metrics}
                rows.append(row)
                (folder / 'summary.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n')
            completed = rows[-len(study['cases']):]
            print(variant['name'], sum(row['success'] for row in completed), '/', len(completed), flush=True)
    plot_summary(rows, study, folder)
    print(f'完成：{len(rows)} 次仿真；汇总 {folder / "summary.json"}', flush=True)


if __name__ == '__main__':
    main()
