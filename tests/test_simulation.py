"""Physical acceptance tests for the nonlinear simulation and input interface."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from cartpole_model import Parameters
from cartpole_simulation import CartPoleSimulator, SolverSettings, simulate_feedback
from run_simulation import read_config


def momentum(result):
    p = result.params
    return ((p.cart_mass + p.pole_mass) * result.velocity
            + p.pole_mass * p.length * result.angular_velocity * np.cos(result.angle))


class SimulationTests(unittest.TestCase):
    def test_exact_upright_equilibrium_and_final_time(self):
        result = simulate_feedback(controller=lambda t, x: 0.0, duration=0.25, output_dt=0.1)
        np.testing.assert_allclose(result.time, [0, 0.1, 0.2, 0.25], atol=1e-15)
        np.testing.assert_array_equal(result.state, np.zeros((4, 4)))

    def test_unforced_energy_and_horizontal_momentum(self):
        result = simulate_feedback(controller=lambda t, x: 0.0, params=Parameters(0.8, 0.2, 0.7),
                          initial_state=[0.1, 0.2, 0.3, -0.4], duration=6)
        self.assertLess(np.max(np.abs(result.energy_balance_error)), 1e-8)
        np.testing.assert_allclose(momentum(result), momentum(result)[0], atol=1e-8)

    def test_feedback_impulse_matches_horizontal_momentum_change(self):
        result = simulate_feedback(controller=lambda t, x: -0.5 * x[1],
                                   initial_state=[0, 0.2, 0.1, -0.3],
                                   sample_period=0.03, duration=0.25, output_dt=0.1)
        intervals = np.diff(np.r_[result.control_log[:, 0], result.time[-1]])
        impulse = np.sum(result.control_log[:, -1] * intervals)
        self.assertAlmostEqual(momentum(result)[-1] - momentum(result)[0], impulse, places=10)
        self.assertLess(np.max(np.abs(result.energy_balance_error)), 1e-9)

    def test_step_reset_clears_state_clock_and_work(self):
        engine = CartPoleSimulator(initial_state=[0, 0.2, 0.1, 0])
        state = engine.step(-0.5 * engine.state[1], 0.1)
        state[:] = 100
        self.assertFalse(np.any(engine.state == 100))
        self.assertNotEqual(engine.input_work, 0)
        np.testing.assert_array_equal(engine.reset(), np.zeros(4))
        self.assertEqual(engine.time, 0)
        self.assertEqual(engine.input_work, 0)

    def test_tighter_solver_agrees_with_default(self):
        kwargs = dict(initial_state=[0, 0, 0.2, 0], controller=lambda t, x: -0.2 * x[1], duration=4)
        normal = simulate_feedback(**kwargs)
        refined = simulate_feedback(**kwargs, solver=SolverSettings(max_step=0.005, rtol=1e-11, atol=1e-13))
        np.testing.assert_allclose(normal.state, refined.state, rtol=1e-7, atol=1e-8)

    def test_feedback_csv_output_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            result = simulate_feedback(controller=lambda t, x: -0.5 * x[1],
                                       initial_state=[0, 0.2, 0.1, 0], duration=0.2)
            result.save(folder)
            data = np.genfromtxt(folder / 'trajectory.csv', names=True, delimiter=',')
            np.testing.assert_array_equal(data['angle_rad'], result.angle)
            np.testing.assert_array_equal(data['time_s'], result.time)
            np.testing.assert_allclose(data['angle_deg'], np.rad2deg(result.angle))
            self.assertFalse((folder / 'trajectory.npz').exists())
            self.assertFalse((folder / 'force.csv').exists())
            control = np.loadtxt(folder / 'control.csv', delimiter=',', skiprows=1)
            np.testing.assert_array_equal(control, result.control_log)
            metadata = json.loads((folder / 'metadata.json').read_text())
            self.assertEqual(metadata['force_input']['file'], 'control.csv')

    def test_parameter_and_initial_state_config_interfaces(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'config.json'
            path.write_text(json.dumps({'parameters': {'M': 2, 'm': 0.3, 'L': 0.8},
                                        'initial_state': {'angle_deg': 90, 'angular_velocity_deg_s': 180},
                                        'control': {'sample_period_s': 0.02,
                                                    'file': str(Path(__file__).resolve().parents[1] / 'feedback.py'),
                                                    'factory': 'create_pid_controller'}}))
            _, kwargs = read_config(path)
            self.assertEqual(kwargs['params'], Parameters(2, 0.3, 0.8))
            np.testing.assert_allclose(kwargs['initial_state'], [0, 0, np.pi / 2, np.pi])
            path.write_text('{"parameters": {"mass": 2}, "control": {}}')
            with self.assertRaisesRegex(ValueError, 'Unknown keys in parameters'):
                read_config(path)

    def test_invalid_inputs_fail_explicitly(self):
        for kwargs in [{'duration': 0}, {'output_dt': -1}, {'initial_state': [0, 0]}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                simulate_feedback(controller=lambda t, x: 0.0, **kwargs)
        with self.assertRaises(ValueError):
            CartPoleSimulator().step(1, -0.1)
        for value in [np.nan, np.inf, [1], '1', None, 1j, lambda t: 0]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                CartPoleSimulator().step(value, 0.1)


if __name__ == '__main__':
    unittest.main()
