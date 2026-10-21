"""Continuous nonlinear cart-pole with sampled feedback inputs.

SI units throughout; state order is [p, v, theta, omega]. theta=0 is
upright, positive theta leans right. Feedback laws are supplied externally.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
import json

import numpy as np
from scipy.integrate import solve_ivp

from cartpole_model import DEFAULT_PARAMETERS, Parameters, _state_array, nonlinear_rhs, total_energy


def _positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive.")
    return value


@dataclass(frozen=True)
class SolverSettings:
    max_step: float = 0.01
    rtol: float = 1e-9
    atol: float = 1e-11

    def __post_init__(self):
        for name in ('max_step', 'rtol', 'atol'):
            object.__setattr__(self, name, _positive(getattr(self, name), name))


def _force_value(value):
    raw = np.asarray(value)
    if raw.ndim != 0 or not np.issubdtype(raw.dtype, np.number) or np.iscomplexobj(raw):
        raise ValueError('Force must be one real scalar.')
    u = float(raw)
    if not np.isfinite(u):
        raise ValueError('Force must be finite.')
    return u


def _integrate(t0, t1, augmented_state, u, params, solver):
    def rhs(t, z):
        # Integrate external work alongside the four physical states.
        return np.r_[nonlinear_rhs(t, z[:4], u, params), u * z[1]]

    solution = solve_ivp(rhs, (t0, t1), augmented_state, method='DOP853',
                         max_step=solver.max_step, rtol=solver.rtol,
                         atol=solver.atol, dense_output=True)
    if not solution.success or not np.all(np.isfinite(solution.y)):
        raise RuntimeError(f'Integration failed at t={solution.t[-1]}: {solution.message}')
    return solution


@dataclass
class SimulationResult:
    time: np.ndarray
    state: np.ndarray
    force: np.ndarray
    derivatives: np.ndarray
    energy: np.ndarray
    work: np.ndarray
    params: Parameters
    metadata: dict
    control_log: np.ndarray

    @property
    def position(self):
        return self.state[:, 0]

    @property
    def velocity(self):
        return self.state[:, 1]

    @property
    def angle(self):
        return self.state[:, 2]

    @property
    def angular_velocity(self):
        return self.state[:, 3]

    @property
    def energy_balance_error(self):
        return self.energy - self.energy[0] - self.work

    def save(self, directory):
        """Export full-precision CSVs and metadata without duplicate arrays."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        columns = np.column_stack((self.time, self.force, self.state,
                                   np.rad2deg(self.angle), np.rad2deg(self.angular_velocity),
                                   self.derivatives[:, 1], self.derivatives[:, 3],
                                   self.energy, self.work, self.energy_balance_error))
        header = ('time_s,force_N,position_m,velocity_m_s,angle_rad,angular_velocity_rad_s,'
                  'angle_deg,angular_velocity_deg_s,acceleration_m_s2,angular_acceleration_rad_s2,'
                  'energy_J,input_work_J,energy_balance_error_J')
        np.savetxt(directory / 'trajectory.csv', columns, delimiter=',',
                   header=header, comments='', fmt='%.17g')
        np.savetxt(directory / 'control.csv', self.control_log, delimiter=',',
                   header='time_s,position_m,velocity_m_s,angle_rad,angular_velocity_rad_s,force_N',
                   comments='', fmt='%.17g')
        metadata = dict(self.metadata)
        metadata['parameters'] = asdict(self.params)
        metadata['state_order'] = ['position_m', 'velocity_m_s', 'angle_rad', 'angular_velocity_rad_s']
        metadata['angle_convention'] = 'Unwrapped; zero upright; positive leans right.'
        metadata['model'] = 'Frictionless cart, massless rigid rod, point mass at tip; unlimited track; no collisions.'
        metadata['summary'] = {
            'records': len(self.time), 'final_state': self.state[-1].tolist(),
            'max_abs_energy_balance_error_J': float(np.max(np.abs(self.energy_balance_error))),
            'max_abs_recorded_force_N': float(np.max(np.abs(self.force))),
        }
        (directory / 'metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False,
                                                           allow_nan=False) + '\n', encoding='utf-8')


def simulate_feedback(*, controller, sample_period=0.02, params=DEFAULT_PARAMETERS,
                      initial_state=(0, 0, 0, 0), duration=30.0, output_dt=0.01, solver=None):
    """Call controller(t, state_copy) once per sampling instant, then hold u.

    Sampling starts at t=0 and excludes the terminal time. The last interval
    may be shorter than sample_period. Observation and integration clocks are
    independent of the controller clock. No wall-clock pacing is imposed.
    Each run should receive a newly constructed stateful controller.
    """
    if not callable(controller):
        raise ValueError('controller must be callable: controller(t, state) -> scalar force.')
    duration = _positive(duration, 'duration')
    output_dt = _positive(output_dt, 'output_dt')
    sample_period = _positive(sample_period, 'sample_period')
    solver = solver or SolverSettings()
    initial = _state_array(initial_state).copy()
    tolerance = 8 * np.finfo(float).eps * duration
    samples = np.arange(int(np.floor(duration / sample_period)) + 1) * sample_period
    samples = samples[samples < duration - tolerance]
    boundaries = np.r_[samples, duration]
    times = np.arange(int(np.floor(duration / output_dt)) + 1) * output_dt
    times = np.r_[times[times < duration - tolerance], duration]
    augmented = np.empty((len(times), 5))
    current = np.r_[initial, 0.0]
    augmented[0] = current
    log = []
    evaluations = 0
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        observation = current[:4].copy()
        try:
            answer = controller(float(left), observation.copy())
        except Exception as error:
            raise RuntimeError(f'Feedback callback failed at t={left:.17g} s: {error}') from error
        try:
            u = _force_value(answer)
        except ValueError as error:
            raise ValueError(f'Invalid feedback at t={left:.17g} s: {error}') from error
        log.append(np.r_[left, observation, u])
        solution = _integrate(left, right, current, u, params, solver)
        selected = (times > left) & (times <= right)
        if np.any(selected):
            augmented[selected] = solution.sol(times[selected]).T
        current = solution.y[:, -1].copy()
        evaluations += solution.nfev
    control_log = np.asarray(log)
    # Equal mathematical times can differ by a few ulps between the two grids.
    indices = np.searchsorted(samples, times + tolerance, side='right') - 1
    forces = control_log[indices, -1]
    states = augmented[:, :4]
    derivatives = np.array([nonlinear_rhs(t, x, u, params) for t, x, u in zip(times, states, forces)])
    energy = np.array([total_energy(x, params) for x in states])
    metadata = {
        'duration_s': duration, 'output_dt_s': output_dt, 'initial_state': initial.tolist(),
        'solver': {'method': 'DOP853', **asdict(solver)},
        'control': {'sample_period_s': sample_period, 'updates': len(log),
                    'terminal_policy': 'No update at the final time; hold the last command.',
                    'observation': 'Exact simulated [p, v, theta, omega]; no sensor noise or delay.'},
        'force_input': {'type': 'feedback_zoh', 'file': 'control.csv'},
        'rhs_evaluations': evaluations,
    }
    return SimulationResult(times, states, forces, derivatives, energy, augmented[:, 4],
                            params, metadata, control_log)


class CartPoleSimulator:
    """Single-step interface: observe state, then step(u, dt).

    The supplied force is held constant for dt seconds; only the original
    nonlinear model is integrated. state is returned as a copy.
    """

    def __init__(self, params=DEFAULT_PARAMETERS, initial_state=(0, 0, 0, 0), solver=None):
        self.params = params
        self.solver = solver or SolverSettings()
        self.reset(initial_state)

    @property
    def state(self):
        return self._augmented[:4].copy()

    @property
    def input_work(self):
        return float(self._augmented[4])

    def reset(self, initial_state=(0, 0, 0, 0)):
        self._augmented = np.r_[_state_array(initial_state).copy(), 0.0]
        self.time = 0.0
        return self.state

    def step(self, u, dt):
        dt = _positive(dt, 'dt')
        u = _force_value(u)
        next_time = self.time + dt
        if not np.isfinite(next_time) or next_time <= self.time:
            raise ValueError('dt cannot advance the simulation clock.')
        solution = _integrate(self.time, next_time, self._augmented, u, self.params, self.solver)
        self._augmented = solution.y[:, -1].copy()
        self.time = next_time
        return self.state
