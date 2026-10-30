
"""Tests for the discrete-time LQR controller.

The controller is designed from the linearized model, but must
also stabilize the original nonlinear cart-pole simulation.

Run from the repository root:

    python -m unittest tests/test_lqr.py -v

The optional force limit is checked separately from the unconstrained gain.
"""

import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.linalg import solve_discrete_are
from scipy.signal import cont2discrete

from cartpole_model import Parameters, linear_state_space
from cartpole_simulation import simulate_feedback
from feedback import create_lqr_controller


DEFAULT_Q = [10.0, 1.0, 100.0, 1.0]
DEFAULT_R = 1.0
DEFAULT_TS = 0.02


def lqr_settings(q=None, r=DEFAULT_R):
    """Build the expected controller settings."""
    return {
        "Q_diag": list(DEFAULT_Q if q is None else q),
        "R": float(r),
    }


def reference_lqr_gain(params, settings, sample_period):
    """Compute an independent discrete LQR reference gain.

    This function deliberately does not call the controller under
    test. It provides the mathematical reference for comparison.
    """
    A, B, C, D = linear_state_space(params)

    Ad, Bd, _, _, _ = cont2discrete(
        (A, B, C, D),
        sample_period,
        method="zoh",
    )

    Q = np.diag(settings["Q_diag"])
    R = np.array([[settings["R"]]], dtype=float)

    P = solve_discrete_are(Ad, Bd, Q, R)

    K = np.linalg.solve(
        R + Bd.T @ P @ Bd,
        Bd.T @ P @ Ad,
    )

    return Ad, Bd, K


def make_controller(
    params=None,
    settings=None,
    sample_period=DEFAULT_TS,
):
    """Create a controller using the project's public API."""
    if params is None:
        params = Parameters()

    if settings is None:
        settings = lqr_settings()

    return create_lqr_controller(
        params,
        settings,
        sample_period,
    )


def gain_from_controller(controller):
    """Recover the actual linear feedback gain from basis inputs.

    For u = -Kx, applying each basis state reveals one entry of K.
    """
    basis = np.eye(4)

    return np.array([
        -float(controller(0.0, basis[i]))
        for i in range(4)
    ]).reshape(1, 4)


class LQRTests(unittest.TestCase):

    def test_optional_force_limit_clips_both_signs_without_changing_small_signals(self):
        params = Parameters(1, 1, 0.5, 9.81)
        settings = lqr_settings()
        raw = make_controller(params, settings)
        limited = make_controller(params, {**settings, 'force_limit_N': 20})
        state = np.array([0.01, 0, np.deg2rad(1), 0])
        self.assertAlmostEqual(limited(0, state), raw(0, state), places=12)
        for sign in (-1, 1):
            state = np.array([0, 0, sign * np.pi, 0])
            self.assertGreater(abs(raw(0, state)), 20)
            self.assertAlmostEqual(limited(0, state), np.clip(raw(0, state), -20, 20))

    def test_force_limit_rejects_nonpositive_and_nonfinite_values(self):
        for limit in (0, -1, np.nan, np.inf):
            with self.subTest(limit=limit), self.assertRaisesRegex(ValueError, 'force_limit_N'):
                make_controller(settings={**lqr_settings(), 'force_limit_N': limit})

    def test_zero_state_produces_zero_force(self):
        """Upright equilibrium must produce zero control force."""
        controller = make_controller()

        force = controller(0.0, np.zeros(4))

        self.assertTrue(np.isfinite(force))
        self.assertAlmostEqual(float(force), 0.0, places=12)

    def test_gain_matches_discrete_lqr_reference(self):
        """Controller must implement the discrete optimal feedback."""
        params = Parameters()
        settings = lqr_settings()

        controller = make_controller(params, settings)

        _, _, expected_K = reference_lqr_gain(
            params,
            settings,
            DEFAULT_TS,
        )

        actual_K = gain_from_controller(controller)

        assert_allclose(
            actual_K,
            expected_K,
            rtol=1e-6,
            atol=1e-8,
        )

        # Also verify a general state, not only basis vectors.
        state = np.array([0.2, -0.1, 0.08, 0.15])

        expected_force = float((-expected_K @ state).item())
        actual_force = float(controller(0.4, state))

        self.assertAlmostEqual(
            actual_force,
            expected_force,
            places=7,
        )

    def test_discrete_closed_loop_stability(self):
        """All closed-loop eigenvalues must lie inside unit circle."""
        params = Parameters()
        settings = lqr_settings()

        controller = make_controller(params, settings)
        K = gain_from_controller(controller)

        Ad, Bd, _ = reference_lqr_gain(
            params,
            settings,
            DEFAULT_TS,
        )

        closed_loop = Ad - Bd @ K
        eigenvalues = np.linalg.eigvals(closed_loop)

        self.assertTrue(
            np.all(np.abs(eigenvalues) < 1.0),
            msg=f"Unstable discrete eigenvalues: {eigenvalues}",
        )

    def test_q_r_settings_change_gain_correctly(self):
        """Changing Q and R must produce the corresponding LQR gain."""
        configurations = [
            lqr_settings([10, 1, 100, 1], 1),
            lqr_settings([10, 1, 500, 1], 1),
            lqr_settings([100, 1, 100, 1], 1),
            lqr_settings([10, 1, 100, 1], 10),
        ]

        gains = []

        for settings in configurations:
            with self.subTest(settings=settings):
                controller = make_controller(settings=settings)

                actual_K = gain_from_controller(controller)
                _, _, expected_K = reference_lqr_gain(
                    Parameters(),
                    settings,
                    DEFAULT_TS,
                )

                assert_allclose(
                    actual_K,
                    expected_K,
                    rtol=1e-6,
                    atol=1e-8,
                )

                gains.append(actual_K)

        for other_gain in gains[1:]:
            self.assertFalse(
                np.allclose(gains[0], other_gain),
                msg="Different Q/R choices unexpectedly gave same gain",
            )

    def test_sample_period_changes_discrete_gain(self):
        """Different sample periods require different discrete gains."""
        params = Parameters()
        settings = lqr_settings()

        gains = []

        for Ts in (0.01, 0.02, 0.05):
            with self.subTest(sample_period=Ts):
                controller = make_controller(
                    params,
                    settings,
                    Ts,
                )

                actual_K = gain_from_controller(controller)
                _, _, expected_K = reference_lqr_gain(
                    params,
                    settings,
                    Ts,
                )

                assert_allclose(
                    actual_K,
                    expected_K,
                    rtol=1e-6,
                    atol=1e-8,
                )

                gains.append(actual_K)

        self.assertFalse(np.allclose(gains[0], gains[-1]))

    def test_physical_parameters_change_gain_correctly(self):
        """LQR design must use the supplied cart-pole parameters."""
        settings = lqr_settings()

        parameter_sets = [
            Parameters(),
            Parameters(
                cart_mass=2.0,
                pole_mass=0.2,
                length=0.7,
                gravity=9.81,
            ),
        ]

        gains = []

        for params in parameter_sets:
            with self.subTest(params=params):
                controller = make_controller(
                    params=params,
                    settings=settings,
                )

                actual_K = gain_from_controller(controller)

                _, _, expected_K = reference_lqr_gain(
                    params,
                    settings,
                    DEFAULT_TS,
                )

                assert_allclose(
                    actual_K,
                    expected_K,
                    rtol=1e-6,
                    atol=1e-8,
                )

                gains.append(actual_K)

        self.assertFalse(np.allclose(gains[0], gains[1]))

    def test_nonlinear_cartpole_stabilization(self):
        """LQR should stabilize the nonlinear plant from 5 degrees."""
        controller = make_controller()

        initial_state = [
            0.0,
            0.0,
            np.deg2rad(5.0),
            0.0,
        ]

        result = simulate_feedback(
            controller=controller,
            initial_state=initial_state,
            sample_period=DEFAULT_TS,
            duration=30.0,
            output_dt=0.02,
        )

        self.assertTrue(np.all(np.isfinite(result.state)))
        self.assertTrue(np.all(np.isfinite(result.force)))

        final_state = result.state[-1]

        # Numerical acceptance criteria, not course requirements.
        self.assertLess(abs(final_state[0]), 0.05)
        self.assertLess(abs(final_state[1]), 0.05)
        self.assertLess(abs(final_state[2]), np.deg2rad(0.5))
        self.assertLess(abs(final_state[3]), np.deg2rad(1.0))

        # Check that the state stays close to equilibrium at the end.
        tail = result.state[result.time >= 25.0]

        self.assertLess(
            np.max(np.abs(tail[:, 2])),
            np.deg2rad(0.5),
        )

        self.assertLess(
            np.max(np.abs(tail[:, 0])),
            0.05,
        )

        # The controller must actually apply nonzero force.
        self.assertGreater(
            np.max(np.abs(result.force)),
            1e-5,
        )

    def test_controller_instances_are_independent(self):
        """Creating a new controller must not inherit earlier state."""
        state = np.array([0.1, 0.0, 0.05, -0.1])

        first = make_controller()
        second = make_controller()

        u1 = first(0.0, state)
        u2 = second(0.0, state)

        self.assertAlmostEqual(float(u1), float(u2), places=12)

        # A stateless LQR must use the current state, not call history.
        first(0.02, np.array([0.2, 0.1, -0.1, 0.0]))

        self.assertAlmostEqual(
            float(first(0.04, state)),
            float(second(0.04, state)),
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
