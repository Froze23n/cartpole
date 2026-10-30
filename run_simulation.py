"""JSON-configured nonlinear simulation with sampled feedback plugins."""

import argparse
from datetime import datetime
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys
import uuid
from zoneinfo import ZoneInfo

import numpy as np

from cartpole_model import Parameters
from cartpole_simulation import SolverSettings, simulate_feedback

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = PROJECT_ROOT / 'results'
TIMEZONE = ZoneInfo('Asia/Shanghai')


def _keys(obj, allowed, context):
    if not isinstance(obj, dict):
        raise ValueError(f'{context} must be a JSON object.')
    unknown = set(obj) - set(allowed)
    if unknown:
        raise ValueError(f'Unknown keys in {context}: {sorted(unknown)}')


def _validate_config(config):
    if isinstance(config, dict) and 'force' in config:
        raise ValueError('force 输入配置已移除，请使用 control 配置离散反馈函数。')
    _keys(config, ['parameters', 'initial_state', 'simulation', 'control', 'output'], 'config')
    if 'control' not in config:
        raise ValueError('必须提供 control 配置，以指定离散反馈函数。')
    _keys(config['control'], ['sample_period_s', 'file', 'factory', 'settings'], 'control')


def read_output_settings(config):
    spec = config.get('output', {})
    _keys(spec, ['plot', 'animation'], 'output')
    if 'plot' in spec and not isinstance(spec['plot'], bool):
        raise ValueError('output.plot must be true or false.')
    animation = spec.get('animation', {})
    _keys(animation, ['format', 'fps', 'speed'], 'output.animation')
    fmt, fps, speed = animation.get('format', 'none'), animation.get('fps', 20), animation.get('speed', 1.0)
    if fmt not in ('none', 'mp4', 'gif'):
        raise ValueError('output.animation.format must be none, mp4 or gif.')
    if isinstance(fps, bool) or not isinstance(fps, int) or fps <= 0:
        raise ValueError('output.animation.fps must be a positive integer.')
    speed = float(speed)
    if not np.isfinite(speed) or speed <= 0:
        raise ValueError('output.animation.speed must be finite and positive.')
    return dict(plot=spec.get('plot', True), animation=fmt, fps=fps, speed=speed)


def _resolve_input(config_path, key, value):
    """Prefer an archived input when rerunning a verbatim config snapshot."""
    manifest_path = config_path.parent / 'metadata.json'
    if manifest_path.exists():
        archive = json.loads(manifest_path.read_text(encoding='utf-8')).get('archive', {})
        if archive.get('configuration') == config_path.name and key in archive.get('files', {}):
            return (config_path.parent / archive['files'][key]).resolve()
    return (config_path.parent / value).resolve()


def _archive_run(config_path, started_at):
    """Copy inputs before loading code or integrating, preserving JSON bytes."""
    config_path = config_path.resolve()
    original = config_path.read_bytes()
    config = json.loads(original)
    _validate_config(config)
    read_output_settings(config)
    source = _resolve_input(config_path, 'control.file', config['control']['file'])
    feedback_source = source.read_bytes()
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    output = RESULTS_ROOT / started_at.strftime('%Y%m%d-%H%M%S-%f')
    output.mkdir()  # Never merge with or overwrite another run.
    (output / 'config.json').write_bytes(original)
    (output / 'feedback_source.py').write_bytes(feedback_source)
    manifest = {
        'run': {'started_at': started_at.isoformat(), 'timezone': 'Asia/Shanghai', 'status': 'running'},
        'archive': {'configuration': 'config.json',
                    'files': {'control.file': 'feedback_source.py'}},
    }
    (output / 'metadata.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return output, manifest


def _load_controller(spec, config_path, params):
    _keys(spec, ['sample_period_s', 'file', 'factory', 'settings'], 'control')
    period = float(spec['sample_period_s'])
    if not np.isfinite(period) or period <= 0:
        raise ValueError('control.sample_period_s must be finite and positive.')
    settings = spec.get('settings', {})
    if not isinstance(settings, dict):
        raise ValueError('control.settings must be a JSON object.')
    source = _resolve_input(config_path, 'control.file', spec['file'])
    factory_name = spec.get('factory', 'create_controller')
    # Fresh module and factory instance for every run, including stateful controllers.
    module_name = '_cartpole_feedback_' + uuid.uuid4().hex
    module_spec = importlib.util.spec_from_file_location(module_name, source)
    if module_spec is None or module_spec.loader is None:
        raise ValueError(f'Cannot load feedback module: {source}')
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
    try:
        # Execute the exact snapshot without adding bytecode caches to results.
        exec(compile(source.read_bytes(), str(source), 'exec'), module.__dict__)
        factory = getattr(module, factory_name, None)
        if not callable(factory):
            raise ValueError(f'Feedback factory {factory_name!r} is not callable in {source}.')
        controller = factory(parameters=params, settings=settings, sample_period=period)
        if not callable(controller):
            raise ValueError('Feedback factory must return controller(t, state).')
    except Exception as error:
        sys.modules.pop(module_name, None)
        raise ValueError(f'Cannot initialize feedback from {source}: {error}') from error
    return controller, period


def read_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding='utf-8'))
    _validate_config(config)
    read_output_settings(config)
    p = config.get('parameters', {})
    _keys(p, ['M', 'm', 'L', 'g'], 'parameters')
    params = Parameters(cart_mass=p.get('M', 1.0), pole_mass=p.get('m', 0.1),
                        length=p.get('L', 0.5), gravity=p.get('g', 9.81))
    initial = config.get('initial_state', {})
    _keys(initial, ['position_m', 'velocity_m_s', 'angle_deg', 'angle_rad',
                    'angular_velocity_deg_s', 'angular_velocity_rad_s'], 'initial_state')
    if 'angle_deg' in initial and 'angle_rad' in initial:
        raise ValueError('Specify angle_deg or angle_rad, not both.')
    if 'angular_velocity_deg_s' in initial and 'angular_velocity_rad_s' in initial:
        raise ValueError('Specify angular velocity in degrees or radians, not both.')
    x0 = [initial.get('position_m', 0), initial.get('velocity_m_s', 0),
          initial.get('angle_rad', np.deg2rad(initial.get('angle_deg', 0))),
          initial.get('angular_velocity_rad_s', np.deg2rad(initial.get('angular_velocity_deg_s', 0)))]
    settings = config.get('simulation', {})
    _keys(settings, ['duration_s', 'output_dt_s', 'max_step_s', 'rtol', 'atol'], 'simulation')
    solver = SolverSettings(max_step=settings.get('max_step_s', 0.01),
                            rtol=settings.get('rtol', 1e-9), atol=settings.get('atol', 1e-11))
    kwargs = dict(params=params, initial_state=x0, duration=settings.get('duration_s', 30),
                  output_dt=settings.get('output_dt_s', 0.01), solver=solver)
    kwargs['controller'], kwargs['sample_period'] = _load_controller(config['control'], path, params)
    return config, kwargs


def main(argv=None):
    parser = argparse.ArgumentParser(description='JSON 配置的非线性倒立摆仿真；运行选项全部在 JSON 中')
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT / 'configs' / 'pid_02_angle_30deg.json',
                        help='选择 JSON 文件；省略时读取 configs/pid_02_angle_30deg.json')
    args = parser.parse_args(argv)
    started_at = datetime.now(TIMEZONE)
    output = None
    try:
        output, manifest = _archive_run(args.config, started_at)
        config, kwargs = read_config(output / 'config.json')
        options = read_output_settings(config)
        mode = f'离散反馈，Ts={kwargs["sample_period"]} s'
        print(f'开始仿真：{kwargs["duration"]} s，完整非线性模型，{mode}。', flush=True)
        result = simulate_feedback(**kwargs)
        result.metadata.update(manifest)
        result.metadata['versions'] = {'python': sys.version.split()[0], **{
            name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'matplotlib')}}
        result.save(output)
        if options['plot'] or options['animation'] != 'none':
            import matplotlib
            matplotlib.use('Agg')
            from cartpole_visualization import plot_result, animate_result
            if options['plot']:
                plot_result(result, output / 'response.png')
            if options['animation'] != 'none':
                print(f'正在生成 {options["animation"].upper()} 动画……', flush=True)
                animate_result(result, output / f'animation.{options["animation"]}',
                               fps=options['fps'], playback_speed=options['speed'])
        metadata_path = output / 'metadata.json'
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        metadata['run'].update(status='completed', finished_at=datetime.now(TIMEZONE).isoformat())
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        print(f'反馈计算 {len(result.control_log)} 次，已保存 control.csv。')
        print(f'已输出 {len(result.time)} 条状态记录。')
        print(f'能量平衡最大误差：{np.max(np.abs(result.energy_balance_error)):.3e} J')
        print(f'结果目录：{output.resolve()}')
        return 0
    except (ValueError, TypeError, KeyError, OSError, RuntimeError) as error:
        if output is not None:
            metadata_path = output / 'metadata.json'
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
            metadata['run'].update(status='failed', error=str(error))
            metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        parser.exit(2, f'仿真失败：{error}\n')


if __name__ == '__main__':
    raise SystemExit(main())
