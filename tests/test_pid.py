"""PID sign, discrete memory, saturation recovery and nonlinear acceptance."""

from copy import deepcopy
import json
from pathlib import Path
import unittest

import numpy as np

from cartpole_model import Parameters
from cartpole_simulation import simulate_feedback
from feedback import create_pid_controller


class PIDTests(unittest.TestCase):
    def setUp(self):
        config = json.loads((Path(__file__).resolve().parents[1] / 'configs' / 'pid.json').read_text())
        self.settings = config['control']['settings']
        p = config['parameters']
        self.params = Parameters(p['M'], p['m'], p['L'], p['g'])
        self.period = config['control']['sample_period_s']

    def controller(self, settings=None, period=None):
        return create_pid_controller(self.params, self.settings if settings is None else settings,
                                     self.period if period is None else period)

    def test_correct_force_direction_and_equilibrium(self):
        self.assertEqual(self.controller()(0, [0, 0, 0, 0]), 0)
        right = self.controller()(0, [0, 0, np.deg2rad(5), 0])
        left = self.controller()(0, [0, 0, np.deg2rad(-5), 0])
        self.assertGreater(right, 0)
        self.assertEqual(left, -right)

    def test_integral_uses_sample_period_skips_initial_time_and_resets(self):
        settings = deepcopy(self.settings)
        settings['position_pid'].update(kp=0, ki=0, kd=0)
        settings['angle_pid'].update(kp=0, ki=2, kd=0)
        for period in [0.05, 0.1]:
            with self.subTest(period=period):
                controller = self.controller(settings, period)
                self.assertEqual(controller(0, [0, 0, 0.1, 0]), 0)
                self.assertAlmostEqual(controller(period, [0, 0, 0.1, 0]), 0.2 * period)
                self.assertAlmostEqual(controller(2 * period, [0, 0, 0.1, 0]), 0.4 * period)
                self.assertEqual(self.controller(settings, period)(0, [0, 0, 0.1, 0]), 0)

    def test_force_saturation_does_not_leave_inner_integral_wound_up(self):
        settings = deepcopy(self.settings)
        settings['position_pid'].update(kp=0, ki=0, kd=0)
        settings['angle_pid'].update(kp=10, ki=5, kd=0)
        settings['force_limit_N'] = 1
        controller = self.controller(settings)
        for k in range(100):
            self.assertEqual(controller(k * self.period, [0, 0, 1, 0]), 1)
        self.assertEqual(controller(100 * self.period, [0, 0, 0, 0]), 0)
        self.assertLess(controller(101 * self.period, [0, 0, -0.05, 0]), 0)

    def test_inner_force_saturation_also_freezes_outer_integral(self):
        settings = deepcopy(self.settings)
        settings['position_pid'].update(kp=0, ki=1, kd=0)
        settings['angle_pid'].update(kp=100, ki=0, kd=0)
        settings['force_limit_N'] = 1
        controller = self.controller(settings)
        for k in range(100):
            self.assertEqual(controller(k * self.period, [-1, 0, 1, 0]), 1)
        self.assertEqual(controller(100 * self.period, [0, 0, 0, 0]), 0)

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
        self.assertLess(controller(100 * self.period, [0, 0, -0.1, 0]), force)

    def test_derivative_uses_measured_velocity_without_reference_kick(self):
        settings = deepcopy(self.settings)
        settings['position_target_m'] = 1
        settings['angle_pid'].update(kp=0, ki=0, kd=20)
        self.assertEqual(self.controller(settings)(0, [0, 0, 0, 0]), 0)
        self.assertAlmostEqual(self.controller(settings)(0, [0, 0, 0, 0.1]), 2)

    def test_invalid_or_incomplete_pid_settings_fail(self):
        invalid = [{}, {**self.settings, 'typo': 1}]
        for key, value in [('force_limit_N', 0), ('angle_limit_deg', 90),
                           ('position_target_m', float('nan')), ('force_limit_N', True)]:
            invalid.append({**self.settings, key: value})
        for key, value in [('kp', -1), ('kd', '20'), ('ki', float('inf')), ('integral_limit', -1)]:
            spec = deepcopy(self.settings)
            spec['angle_pid'][key] = value
            invalid.append(spec)
        for spec in invalid:
            with self.subTest(settings=spec), self.assertRaises(ValueError):
                self.controller(spec)
        with self.assertRaises(ValueError):
            self.controller(period=0)

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


if __name__ == '__main__':
    unittest.main()
