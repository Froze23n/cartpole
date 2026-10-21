"""Ideal cart-pole model.

State order: [p, p_dot, theta, theta_dot].
p and force u are positive to the right. theta = 0 is upright,
and positive theta tilts the pole to the right.
The rod is massless; its entire pendulum mass is at its tip.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Parameters:
    cart_mass: float = 1.0       # M, kg
    pole_mass: float = 0.1       # m, kg; point mass at the tip
    length: float = 0.5         # l, m; pivot-to-point-mass distance
    gravity: float = 9.81       # g, m/s^2

    def __post_init__(self):
        values = (self.cart_mass, self.pole_mass, self.length, self.gravity)
        if not all(np.isfinite(value) and value > 0 for value in values):
            raise ValueError("Masses, length, and gravity must be finite and positive.")


DEFAULT_PARAMETERS = Parameters()


def _state_array(state):
    state = np.asarray(state, dtype=float)
    if state.shape != (4,) or not np.all(np.isfinite(state)):
        raise ValueError("State must be a finite array with shape (4,).")
    return state


def nonlinear_rhs(t, state, u, params=DEFAULT_PARAMETERS):
    """Return d(state)/dt for a given applied horizontal force u (N).

    Compatible with solve_ivp via args=(u, params).
    For sampled control, keep u fixed while integrating each sampling interval.
    There is no saturation, friction, rail collision, or controller in this model.
    """
    _, velocity, theta, omega = _state_array(state)
    u = float(u)
    if not np.isfinite(u):
        raise ValueError("Applied force must be finite.")

    M, m, l, g = (
        params.cart_mass, params.pole_mass, params.length, params.gravity
    )
    sine, cosine = np.sin(theta), np.cos(theta)
    denominator = M + m * sine**2
    acceleration = (
        u + m * l * omega**2 * sine - m * g * sine * cosine
    ) / denominator
    angular_acceleration = (
        (M + m) * g * sine - u * cosine
        - m * l * omega**2 * sine * cosine
    ) / (l * denominator)
    return np.array([velocity, acceleration, omega, angular_acceleration])


def linear_state_space(params=DEFAULT_PARAMETERS):
    """Return A, B, C, D linearized at the stationary upright equilibrium.

    Shapes: A (4, 4), B (4, 1), C (2, 4), D (2, 1).
    Outputs are [cart position p, pole angle theta].
    """
    M, m, l, g = (
        params.cart_mass, params.pole_mass, params.length, params.gravity
    )
    A = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, -m * g / M, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, (M + m) * g / (M * l), 0.0],
    ])
    B = np.array([[0.0], [1.0 / M], [0.0], [-1.0 / (M * l)]])
    C = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    D = np.zeros((2, 1))
    return A, B, C, D


def tip_position(state, params=DEFAULT_PARAMETERS):
    """Return the point-mass coordinates [horizontal, vertical] in metres.

    The pivot is at [p, 0]; vertical is positive upward.
    """
    position, _, theta, _ = _state_array(state)
    return np.array([
        position + params.length * np.sin(theta),
        params.length * np.cos(theta),
    ])


def total_energy(state, params=DEFAULT_PARAMETERS):
    """Mechanical energy T + V in joules, with V = 0 at pivot height."""
    _, velocity, theta, omega = _state_array(state)
    M, m, l, g = (
        params.cart_mass, params.pole_mass, params.length, params.gravity
    )
    kinetic = (
        0.5 * (M + m) * velocity**2
        + m * l * velocity * omega * np.cos(theta)
        + 0.5 * m * l**2 * omega**2
    )
    potential = m * g * l * np.cos(theta)
    return float(kinetic + potential)


def transfer_functions(params=DEFAULT_PARAMETERS):
    """Zero-initial-condition continuous linear transfer functions.

    Each value is (numerator, denominator), in descending powers of s.
    The angle transfer function alone does not describe cart-position modes.
    """
    M, m, l, g = (
        params.cart_mass, params.pole_mass, params.length, params.gravity
    )
    return {
        "force_to_position": (
            np.array([l, 0.0, -g]),
            np.array([M * l, 0.0, -(M + m) * g, 0.0, 0.0]),
        ),
        "force_to_angle": (
            np.array([-1.0]),
            np.array([M * l, 0.0, -(M + m) * g]),
        ),
    }

