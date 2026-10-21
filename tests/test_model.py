"""Physical and mathematical consistency checks for the cart-pole model."""

import unittest

import numpy as np

from cartpole_model import (
    Parameters, linear_state_space, nonlinear_rhs, tip_position,
    total_energy, transfer_functions,
)


class ModelTests(unittest.TestCase):
    def test_stationary_equilibria(self):
        for position in (0.0, 0.7):
            for angle in (0.0, np.pi):
                state = [position, 0.0, angle, 0.0]
                np.testing.assert_allclose(
                    nonlinear_rhs(0.0, state, 0.0), np.zeros(4), atol=1e-12
                )

    def test_geometry_and_force_direction(self):
        params = Parameters()
        tip = tip_position([0.2, 0.0, np.pi / 2, 0.0], params)
        np.testing.assert_allclose(tip, [0.2 + params.length, 0.0], atol=1e-12)
        derivative = nonlinear_rhs(0.0, np.zeros(4), 1.0, params)
        self.assertGreater(derivative[1], 0.0)
        self.assertLess(derivative[3], 0.0)
        falling = nonlinear_rhs(0.0, [0.0, 0.0, 0.05, 0.0], 0.0, params)
        self.assertGreater(falling[3], 0.0)

    def test_linearization_matches_numerical_jacobian(self):
        # Check both default and non-default parameters.
        for params in (Parameters(), Parameters(0.8, 0.3, 0.7, 9.8)):
            A, B, _, _ = linear_state_space(params)
            step = 1e-6
            numerical_A = np.empty((4, 4))
            for column in range(4):
                delta = np.eye(4)[column] * step
                numerical_A[:, column] = (
                    nonlinear_rhs(0.0, delta, 0.0, params)
                    - nonlinear_rhs(0.0, -delta, 0.0, params)
                ) / (2 * step)
            numerical_B = (
                nonlinear_rhs(0.0, np.zeros(4), step, params)
                - nonlinear_rhs(0.0, np.zeros(4), -step, params)
            ) / (2 * step)
            np.testing.assert_allclose(A, numerical_A, rtol=1e-8, atol=1e-10)
            np.testing.assert_allclose(B[:, 0], numerical_B, rtol=1e-8, atol=1e-10)

    def test_energy_rate_equals_external_power(self):
        # In the frictionless system, dE/dt must equal u * cart_velocity.
        states = ([0.2, -0.3, 0.4, 0.6], [-1.0, 0.8, -0.7, -1.2])
        params = Parameters(0.8, 0.3, 0.7, 9.8)
        step = 1e-6
        for values in states:
            state = np.array(values)
            gradient = np.array([
                (total_energy(state + step * axis, params)
                 - total_energy(state - step * axis, params)) / (2 * step)
                for axis in np.eye(4)
            ])
            for force in (0.0, 1.3, -2.1):
                energy_rate = gradient @ nonlinear_rhs(0.0, state, force, params)
                np.testing.assert_allclose(
                    energy_rate, force * state[1], rtol=1e-7, atol=1e-7
                )

    def test_transfer_functions_match_state_space(self):
        params = Parameters(0.8, 0.3, 0.7, 9.8)
        A, B, C, D = linear_state_space(params)
        functions = transfer_functions(params)
        for s in (2.0 + 3.0j, 4.0 + 0.7j, 8.0 + 2.0j):
            state_space_response = (C @ np.linalg.solve(s * np.eye(4) - A, B) + D)[:, 0]
            for index, key in enumerate(("force_to_position", "force_to_angle")):
                numerator, denominator = functions[key]
                transfer_response = np.polyval(numerator, s) / np.polyval(denominator, s)
                np.testing.assert_allclose(
                    transfer_response, state_space_response[index],
                    rtol=1e-12, atol=1e-12,
                )

    def test_full_model_controllability_and_observability(self):
        A, B, C, _ = linear_state_space()
        controllability = np.hstack([np.linalg.matrix_power(A, k) @ B for k in range(4)])
        observability = np.vstack([C @ np.linalg.matrix_power(A, k) for k in range(4)])
        self.assertEqual(np.linalg.matrix_rank(controllability), 4)
        self.assertEqual(np.linalg.matrix_rank(observability), 4)


if __name__ == "__main__":
    unittest.main()
