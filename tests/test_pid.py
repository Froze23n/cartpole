"""PID sign, measured rates, bounded integrals and nonlinear acceptance."""

from copy import deepcopy
import inspect
import json
from pathlib import Path
import unittest

import numpy as np
from scipy.signal import cont2discrete

from cartpole_model import Parameters, linear_state_space
from cartpole_simulation import CartPoleSimulator, simulate_feedback
from feedback import create_pid_controller, tune_angle_pid, tune_position_pid


def pid_factory_for_test(use_tuning):
    """在内存副本中切换注释，不改工作区；两种用户选择都可测试。"""
    source = inspect.getsource(create_pid_controller)
    for statement, enabled in [
        ("position_pid = settings['position_pid']", not use_tuning),
        ("angle_pid = settings['angle_pid']", not use_tuning),
        ('angle_pid = tune_angle_pid(parameters, settings)', use_tuning),
        ('position_pid = tune_position_pid(parameters, settings, angle_pid, sample_period)', use_tuning),
    ]:
        source = source.replace('    # ' + statement, '    ' + statement)
        if not enabled:
            source = source.replace('    ' + statement, '    # ' + statement)
    namespace = create_pid_controller.__globals__.copy()
    exec(compile(source, '<PID parameter selection>', 'exec'), namespace)
    return namespace['create_pid_controller']


class PIDTests(unittest.TestCase):
    def setUp(self):
        config = json.loads((Path(__file__).resolve().parents[1] / 'configs' / 'pid_02_angle_30deg.json').read_text())
        self.settings = config['control']['settings']
        p = config['parameters']
        self.params = Parameters(p['M'], p['m'], p['L'], p['g'])
        self.period = config['control']['sample_period_s']
        self.manual_factory = pid_factory_for_test(use_tuning=False)

    def controller(self, settings=None, period=None):
        return self.manual_factory(self.params, self.settings if settings is None else settings,
                                   self.period if period is None else period)

    def test_correct_force_direction_and_equilibrium(self):
        self.assertEqual(self.controller()(0, [0, 0, 0, 0]), 0)
        right = self.controller()(0, [0, 0, np.deg2rad(5), 0])
        left = self.controller()(0, [0, 0, np.deg2rad(-5), 0])
        self.assertGreater(right, 0)
        self.assertEqual(left, -right)

    def test_integral_updates_after_output_and_resets(self):
        settings = deepcopy(self.settings)
        settings['position_pid'].update(kp=0, ki=0, kd=0)
        settings['angle_pid'].update(kp=0, ki=2, kd=0)
        for period in [0.05, 0.1]:
            with self.subTest(period=period):
                controller = self.controller(settings, period)
                self.assertEqual(controller(0, [0, 0, 0.1, 0]), 0)
                self.assertAlmostEqual(controller(period, [0, 0, 0.3, 0]), 0.2 * period)
                self.assertAlmostEqual(controller(2 * period, [0, 0, -0.2, 0]), 0.8 * period)
                self.assertAlmostEqual(controller(3 * period, [0, 0, 0, 0]), 0.4 * period)
                self.assertEqual(self.controller(settings, period)(0, [0, 0, 0.1, 0]), 0)

    def test_force_saturation_keeps_a_bounded_integral_contribution(self):
        settings = deepcopy(self.settings)
        settings['position_pid'].update(kp=0, ki=0, kd=0)
        settings['angle_pid'].update(kp=10, ki=5, kd=0, integral_limit=0.25)
        settings['force_limit_N'] = 1
        controller = self.controller(settings)
        for k in range(100):
            self.assertEqual(controller(k * self.period, [0, 0, 1, 0]), 1)
        # 积分限幅不等于饱和时冻结积分：误差归零后仍保留有限的 I 项。
        self.assertAlmostEqual(controller(100 * self.period, [0, 0, 0, 0]), 0.25)
        self.assertLess(controller(101 * self.period, [0, 0, -0.05, 0]), 0)

    def test_position_integral_uses_sample_period_and_has_an_angle_limit(self):
        settings = deepcopy(self.settings)
        settings['position_pid'].update(kp=0, ki=0.1, kd=0, integral_limit=0.04)
        settings['angle_pid'].update(kp=1, ki=0, kd=0)
        for period in [0.05, 0.1]:
            with self.subTest(period=period):
                controller = self.controller(settings, period)
                self.assertEqual(controller(0, [-1, 0, 0, 0]), 0)
                self.assertAlmostEqual(controller(period, [-2, 0, 0, 0]), -0.1 * period)
                self.assertAlmostEqual(controller(2 * period, [0, 0, 0, 0]), -0.3 * period)
                for k in range(3, 100):
                    force = controller(k * period, [-1, 0, 0, 0])
                    self.assertLessEqual(abs(force), 0.04)
                self.assertAlmostEqual(force, -0.04)
                self.assertEqual(self.controller(settings, period)(0, [-1, 0, 0, 0]), 0)

    def test_angle_reference_and_integral_contribution_limits(self):
        settings = deepcopy(self.settings)
        settings['position_pid'].update(kp=100, ki=0, kd=0)
        settings['angle_pid'].update(kp=1, ki=0, kd=0)
        # With a unit inner P gain, force numerically exposes the limited reference.
        self.assertAlmostEqual(self.controller(settings)(0, [10, 0, 0, 0]),
                               np.deg2rad(settings['angle_limit_deg']))
        settings['position_pid'].update(kp=0, ki=0, kd=0)
        settings['angle_pid'].update(kp=0, ki=10, kd=0, integral_limit=0.25)
        controller = self.controller(settings)
        for k in range(100):
            force = controller(k * self.period, [0, 0, 0.1, 0])
            self.assertLessEqual(abs(force), 0.25)
        self.assertAlmostEqual(force, 0.25)
        self.assertAlmostEqual(controller(100 * self.period, [0, 0, -0.1, 0]), force)
        self.assertLess(controller(101 * self.period, [0, 0, -0.1, 0]), force)

    def test_derivatives_use_current_rates_even_at_initial_time(self):
        for period in [0.05, 0.1]:
            with self.subTest(period=period):
                settings = deepcopy(self.settings)
                settings['position_pid'].update(kp=0, ki=0, kd=0.2)
                settings['angle_pid'].update(kp=1, ki=0, kd=0)
                controller = self.controller(settings, period)
                self.assertAlmostEqual(controller(0, [0, 0.1, 0, 0]), 0.02)
                self.assertAlmostEqual(controller(period, [0, -0.1, 0, 0]), -0.02)
                # 位置变化不会被额外差分；D 只使用这次传入的 v。
                self.assertEqual(controller(2 * period, [0.01, 0, 0, 0]), 0)

                settings['position_pid'].update(kp=0, ki=0, kd=0)
                settings['angle_pid'].update(kp=0, ki=0, kd=2)
                controller = self.controller(settings, period)
                self.assertAlmostEqual(controller(0, [0, 0, 0.05, 0.1]), 0.2)
                self.assertAlmostEqual(controller(period, [0, 0, 0.05, -0.1]), -0.2)
                self.assertEqual(controller(2 * period, [0, 0, 0.06, 0]), 0)
                self.assertAlmostEqual(self.controller(settings, period)(0, [0, 0, 0.05, 0.1]), 0.2)

    def test_angle_derivative_ignores_changes_in_outer_reference(self):
        settings = deepcopy(self.settings)
        settings['position_target_m'] = 1
        settings['position_pid'].update(kp=0.1, ki=0, kd=0)
        settings['angle_pid'].update(kp=0, ki=0, kd=20)
        controller = self.controller(settings)
        self.assertEqual(controller(0, [0, 0, 0, 0]), 0)
        # 外环目标倾角变化、omega 不变，不会引起内环 D 的额外变化。
        self.assertEqual(controller(self.period, [0.5, 0, 0, 0]), 0)
        self.assertAlmostEqual(controller(2 * self.period, [0.5, 0, 0, 0.1]), 2)

    def test_scenario_configs_control_variables_and_output_settings(self):
        from run_simulation import read_config

        folder = Path(__file__).resolve().parents[1] / 'configs'
        paths = sorted(folder.glob('pid_[0-9][0-9]_*.json'))
        self.assertEqual(len(paths), 8)
        common = None
        initial_states = []
        for path in paths:
            with self.subTest(config=path.name):
                config, kwargs = read_config(path)
                initial_states.append(tuple(kwargs['initial_state']))
                config.pop('initial_state')
                if common is None:
                    common = config
                self.assertEqual(config, common)
                self.assertEqual(kwargs['duration'], 20)
                self.assertTrue(config['output']['plot'])
                self.assertEqual(config['output']['animation']['format'], 'mp4')
                self.assertEqual(config['control']['factory'], 'create_pid_controller')
        self.assertEqual(len(set(initial_states)), len(paths))
        self.assertNotIn((0, 0, 0, 0), initial_states)
        self.assertEqual(sum(0 < abs(state[2]) < np.deg2rad(10) for state in initial_states), 1)
        self.assertIn((0, 0, np.deg2rad(60), 0), initial_states)
        self.assertIn((0, 0, np.pi, 0), initial_states)
        self.assertIn((0, 2, 0, 0), initial_states)
        self.assertIn((0, 0, 0, np.pi), initial_states)
        for state in initial_states:
            self.assertEqual(np.count_nonzero(state), 1)

    def test_nonlinear_balance_and_cart_position_recovery(self):
        # Representative acceptance cases, not a global region-of-attraction claim.
        for position, angle_deg, target in [(0, 5, 0), (0, -5, 0), (0.3, 0, 0),
                                            (0.3, 10, 0), (0, 0, 0.3)]:
            with self.subTest(position=position, angle_deg=angle_deg, target=target):
                settings = deepcopy(self.settings)
                settings['position_target_m'] = target
                result = simulate_feedback(controller=self.controller(settings), params=self.params,
                                           sample_period=self.period, duration=10,
                                           initial_state=[position, 0, np.deg2rad(angle_deg), 0])
                self.assertLess(abs(result.position[-1] - target), 0.01)
                self.assertLess(abs(result.velocity[-1]), 0.01)
                self.assertLess(abs(np.rad2deg(result.angle[-1])), 0.1)
                self.assertLess(abs(np.rad2deg(result.angular_velocity[-1])), 0.1)
                self.assertLess(np.max(np.abs(np.rad2deg(result.angle))), 15)
                self.assertLessEqual(np.max(np.abs(result.force)), settings['force_limit_N'])
                self.assertLess(np.max(np.abs(result.energy_balance_error)), 1e-8)

    def tuned_settings(self, params=None, period=None):
        params = self.params if params is None else params
        period = self.period if period is None else period
        settings = deepcopy(self.settings)
        settings['angle_pid'] = tune_angle_pid(params, self.settings)
        settings['position_pid'] = tune_position_pid(
            params, self.settings, settings['angle_pid'], period)
        return settings

    def test_tuning_places_all_five_continuous_poles_for_different_models(self):
        original = deepcopy(self.settings)
        for params in [self.params, Parameters(1, 0.1, 0.5), Parameters(2, 0.3, 0.8, 5)]:
            with self.subTest(params=params):
                settings = self.tuned_settings(params)
                A, B, _, _ = linear_state_space(params)
                ka = settings['angle_pid']['kp']
                ki = settings['position_pid']['ki']
                # 从实际控制函数取得四个状态的反馈系数，避免另写一遍增益公式。
                force_feedback = np.array([
                    self.manual_factory(params, settings, self.period)(0, state) / 1e-7
                    for state in np.eye(4) * 1e-7
                ]).reshape(1, 4)
                closed_loop = np.block([
                    [A + B @ force_feedback, -ka * B],
                    [np.array([[-ki, 0, 0, 0]]), np.zeros((1, 1))],
                ])
                np.testing.assert_allclose(np.poly(closed_loop), np.poly(settings['tuning_poles']),
                                           rtol=1e-10, atol=1e-10)
                self.assertEqual(settings['angle_pid']['ki'], 0)
                for loop in ['angle_pid', 'position_pid']:
                    self.assertEqual(settings[loop]['integral_limit'], original[loop]['integral_limit'])
        self.assertEqual(self.settings, original)

    def test_comment_switch_selects_json_or_formula_gains(self):
        state = [0, 0, np.deg2rad(5), 0]
        settings = deepcopy(self.settings)
        settings.pop('tuning_poles')
        manual_force = self.controller(settings)(0, state)
        self.assertAlmostEqual(manual_force, settings['angle_pid']['kp'] * state[2])
        formula_factory = pid_factory_for_test(use_tuning=True)
        formula_force = formula_factory(self.params, self.settings, self.period)(0, state)
        self.assertAlmostEqual(formula_force, self.controller(self.tuned_settings())(0, state))
        self.assertNotAlmostEqual(formula_force, manual_force)

    def test_default_factory_uses_poles_instead_of_baseline_gains(self):
        state = [0, 0, np.deg2rad(5), 0]
        settings = deepcopy(self.settings)
        expected = create_pid_controller(self.params, settings, self.period)(0, state)
        for loop in ['position_pid', 'angle_pid']:
            settings[loop].update(kp=0, ki=0, kd=0)
        actual = create_pid_controller(self.params, settings, self.period)(0, state)
        self.assertAlmostEqual(actual, expected)
        settings['tuning_poles'] = [-1, -1.5, -2, -5, -6]
        changed = create_pid_controller(self.params, settings, self.period)(0, state)
        self.assertNotAlmostEqual(changed, expected)

    def test_tuning_rejects_invalid_targets_and_unstable_sampling(self):
        for poles in [[-1, -2], [-1, -2, -3, -4, 0], [-1, -2, -3, -4, 1],
                      [-1, -2, -3, -4, float('nan')]]:
            with self.subTest(poles=poles):
                settings = deepcopy(self.settings)
                settings['tuning_poles'] = poles
                with self.assertRaisesRegex(ValueError, '5 个有限负实数'):
                    tune_angle_pid(self.params, settings)
        with self.assertRaisesRegex(ValueError, '离散闭环不稳定'):
            self.tuned_settings(period=0.2)
        angle_pid = tune_angle_pid(self.params, self.settings)
        angle_pid['ki'] = 1
        with self.assertRaisesRegex(ValueError, 'ki=0'):
            tune_position_pid(self.params, self.settings, angle_pid, self.period)

    def test_tuned_discrete_model_matches_actual_sampled_nonlinear_step(self):
        settings = self.tuned_settings()
        Ts = self.period
        kp, ki, kd = (settings['position_pid'][key] for key in ('kp', 'ki', 'kd'))
        ka, da = (settings['angle_pid'][key] for key in ('kp', 'kd'))
        Ad, Bd, _, _, _ = cont2discrete(linear_state_space(self.params), Ts, method='zoh')
        expected = np.block([
            [Ad + Bd @ np.array([[ka * kp, ka * kd, ka, da]]), -ka * Bd],
            [np.array([[-ki * Ts, 0, 0, 0]]), np.ones((1, 1))],
        ])

        def step(augmented_state):
            state, integral = augmented_state[:4], augmented_state[4]
            controller = self.controller(settings)
            # 预先写入一个微小积分，再从指定物理状态执行真实采样和保持。
            controller(-Ts, [-integral / (ki * Ts), 0, 0, 0])
            force = controller(0, state)
            simulator = CartPoleSimulator(params=self.params, initial_state=state)
            simulator.step(force, Ts)
            return np.r_[simulator.state, integral - ki * Ts * state[0]]

        epsilon = 1e-6
        measured = np.column_stack([(step(d) - step(-d)) / (2 * epsilon)
                                    for d in np.eye(5) * epsilon])
        np.testing.assert_allclose(measured, expected, rtol=1e-7, atol=1e-9)
        self.assertLess(np.max(np.abs(np.linalg.eigvals(measured))), 1)

    def test_tuned_pid_recovers_representative_nonlinear_initial_states(self):
        settings = self.tuned_settings()
        cases = [[0, 0, 5, 0], [0, 0, -5, 0], [0, 0, 30, 0],
                 [0.3, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 30]]
        for case in cases:
            with self.subTest(initial=case):
                state = np.array(case, dtype=float)
                state[2:] = np.deg2rad(state[2:])
                result = simulate_feedback(controller=self.controller(settings), params=self.params,
                                           sample_period=self.period, duration=20, initial_state=state)
                tail = np.abs(result.state[result.time >= 15])
                self.assertTrue(np.all(tail[:, :2] < 0.01))
                self.assertTrue(np.all(tail[:, 2:] < np.deg2rad(0.1)))
                self.assertLessEqual(np.max(np.abs(result.force)), settings['force_limit_N'])
                self.assertLess(np.max(np.abs(result.energy_balance_error)), 1e-8)


if __name__ == '__main__':
    unittest.main()
