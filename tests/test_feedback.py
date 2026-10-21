"""Acceptance tests for sampled feedback timing, state observations and JSON."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from cartpole_simulation import CartPoleSimulator, SolverSettings, simulate_feedback
from run_simulation import main, read_config, read_output_settings


class FeedbackTests(unittest.TestCase):
    def test_callback_once_per_sample_zoh_and_partial_final_interval(self):
        calls = []

        def controller(t, state):
            calls.append(t)
            return len(calls)

        result = simulate_feedback(controller=controller, sample_period=0.1, duration=0.25,
                                   output_dt=0.025, solver=SolverSettings(max_step=0.003))
        np.testing.assert_allclose(calls, [0, 0.1, 0.2])
        np.testing.assert_array_equal(result.force, [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3])
        p = result.params
        momentum = ((p.cart_mass + p.pole_mass) * result.velocity[-1]
                    + p.pole_mass * p.length * result.angular_velocity[-1] * np.cos(result.angle[-1]))
        self.assertAlmostEqual(momentum, 0.45, places=9)
        self.assertLess(np.max(np.abs(result.energy_balance_error)), 1e-9)

    def test_feedback_uses_current_state_and_matches_external_step_loop(self):
        x0 = [0.1, 0.2, 0.1, -0.2]
        result = simulate_feedback(controller=lambda t, x: -2 * x[1], initial_state=x0,
                                   duration=0.4, sample_period=0.1, output_dt=0.1)
        engine = CartPoleSimulator(initial_state=x0)
        states = [engine.state]
        for _ in range(4):
            states.append(engine.step(-2 * engine.state[1], 0.1))
        np.testing.assert_allclose(result.state, states, rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(result.control_log[:, -1], -2 * result.control_log[:, 2])
        self.assertGreater(np.ptp(result.control_log[:, -1]), 0.01)

    def test_callback_cannot_mutate_the_actual_plant_state(self):
        def malicious_mutation(t, x):
            x[:] = 100
            return 0.0
        result = simulate_feedback(controller=malicious_mutation, duration=0.1)
        np.testing.assert_array_equal(result.state, np.zeros_like(result.state))

    def test_independent_output_clock_and_no_update_at_terminal_time(self):
        calls = []
        result = simulate_feedback(controller=lambda t, x: calls.append(t) or 0.0,
                                   sample_period=0.03, duration=0.3, output_dt=0.2)
        self.assertEqual(len(calls), 10)
        self.assertLess(calls[-1], 0.3)
        np.testing.assert_allclose(result.time, [0, 0.2, 0.3])
        tiny = simulate_feedback(controller=lambda t, x: 1.0, sample_period=1, duration=0.01)
        self.assertEqual(len(tiny.control_log), 1)

    def test_observation_spacing_does_not_change_the_feedback_trajectory(self):
        kwargs = dict(controller=lambda t, x: -0.5 * x[1], initial_state=[0, 0, 0.2, 0],
                      duration=0.5, sample_period=0.03)
        coarse = simulate_feedback(**kwargs, output_dt=0.1)
        dense = simulate_feedback(**kwargs, output_dt=0.01)
        np.testing.assert_allclose(coarse.control_log, dense.control_log, rtol=0, atol=0)
        np.testing.assert_allclose(coarse.state[-1], dense.state[-1], rtol=0, atol=0)

    def test_invalid_callback_output_and_period_fail(self):
        for value in [np.nan, np.inf, [1], '1', None, 1j]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                simulate_feedback(controller=lambda t, x: value, duration=0.1)
        with self.assertRaises(ValueError):
            simulate_feedback(controller=lambda t, x: 0, sample_period=0)

    def test_json_plugin_state_reset_and_control_log_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / 'plugin.py').write_text(
                'def create_controller(parameters, settings, sample_period):\n'
                '    calls = 0\n'
                '    def compute(t, state):\n'
                '        nonlocal calls\n'
                '        calls += 1\n'
                '        return settings["gain"] * calls\n'
                '    return compute\n')
            config = {'simulation': {'duration_s': 0.1},
                      'control': {'sample_period_s': 0.05, 'file': 'plugin.py',
                                  'factory': 'create_controller', 'settings': {'gain': 0.1}},
                      'output': {'plot': False, 'animation': {'format': 'none', 'fps': 12, 'speed': 2}}}
            path = folder / 'config.json'
            path.write_text(json.dumps(config))
            _, one = read_config(path)
            _, two = read_config(path)
            first = simulate_feedback(**one)
            second = simulate_feedback(**two)
            np.testing.assert_array_equal(first.control_log[:, -1], [0.1, 0.2])
            np.testing.assert_array_equal(first.state, second.state)
            with patch('run_simulation.RESULTS_ROOT', folder / 'results'):
                self.assertEqual(main(['--config', str(path)]), 0)
            output, = (folder / 'results').iterdir()
            self.assertTrue((output / 'control.csv').is_file())
            self.assertTrue((output / 'feedback_source.py').is_file())
            self.assertFalse((output / 'response.png').exists())
            table = np.genfromtxt(output / 'control.csv', names=True, delimiter=',')
            np.testing.assert_allclose(table['force_N'], [0.1, 0.2])
            settings = read_output_settings(config)
            self.assertNotIn('directory', settings)
            self.assertEqual(settings['fps'], 12)
            self.assertEqual(settings['speed'], 2)

    def test_missing_feedback_and_invalid_output_configuration_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bad.json'
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError, '必须提供 control'):
                read_config(path)
            for output in [{'animation': {'fps': 0}}, {'animation': {'format': 'avi'}},
                           {'plot': 'false'}, {'directory': ''}, {'animation': {'speed': float('nan')}}]:
                path.write_text(json.dumps({'control': {}, 'output': output}))
                with self.subTest(output=output), self.assertRaisesRegex(ValueError, 'output'):
                    read_config(path)


if __name__ == '__main__':
    unittest.main()
