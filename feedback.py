"""离散反馈控制器：串级 PID、离散 LQR 已实现；MPC 保留空壳。

工厂参数：
- parameters：模型参数对象，cart_mass=M [kg]、pole_mass=m [kg]、
  length=L [m]、gravity=g [m/s²]。PID 增益由 JSON 给定，不自动随模型改变。
- settings：JSON 的 control.settings；PID 字段和单位见工厂函数说明。
- sample_period：离散反馈周期 Ts [s]，由平台负责按此周期调用。

控制函数参数：t 为仿真时间 [s]，state 为当前状态副本 [p,v,theta,omega]，
单位分别为 [m,m/s,rad,rad/s]；theta=0 向上，正角向右倾斜。
返回小车水平力 u [N]，向右为正；平台在下一次更新前保持该力。
每次仿真须调用工厂创建新实例，避免继承上次的积分状态。
"""

import numpy as np
from scipy.signal import cont2discrete
from scipy.linalg import solve_discrete_are

from cartpole_model import linear_state_space


def _require_keys(spec, keys, name):
    if not isinstance(spec, dict):
        raise ValueError(f'{name} must be a JSON object.')
    missing, unknown = set(keys) - spec.keys(), spec.keys() - set(keys)
    if missing or unknown:
        raise ValueError(f'{name}: missing keys {sorted(missing)}, unknown keys {sorted(unknown)}.')


def _number(value, name, minimum=None, strictly_positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f'{name} must be a finite real number.')
    value = float(value)
    if (minimum is not None and value < minimum) or (strictly_positive and value <= 0):
        raise ValueError(f'{name} is outside its allowed range.')
    return value


class _PIDLoop:
    """单环并联式 PID；积分状态存的是 I 项的输出贡献，而不是误差累积值。"""

    def __init__(self, spec, limit, name):
        _require_keys(spec, ('kp', 'ki', 'kd', 'integral_limit'), name)
        self.kp, self.ki, self.kd = (
            _number(spec[key], f'{name}.{key}', minimum=0) for key in ('kp', 'ki', 'kd'))
        self.integral_limit = _number(spec['integral_limit'], f'{name}.integral_limit', minimum=0)
        self.limit = limit
        self.integral = 0.0

    def preview(self, error, measured_error_rate, dt):
        # 后向矩形积分；D 项直接使用已测得的速度，不差分误差/目标值。
        candidate = float(np.clip(self.integral + self.ki * dt * error,
                                  -self.integral_limit, self.integral_limit))
        pd = self.kp * error + self.kd * measured_error_rate
        raw = pd + candidate
        change = candidate - self.integral
        # 条件积分：饱和且积分会进一步推向饱和时冻结；反向卸载仍被允许。
        if (raw > self.limit and change > 0) or (raw < -self.limit and change < 0):
            candidate = self.integral
            raw = pd + candidate
        return float(np.clip(raw, -self.limit, self.limit)), candidate, abs(raw) > self.limit


def create_pid_controller(parameters, settings, sample_period):
    """位置外环 + 角度内环，所有可调参数显式来自 JSON。

    settings:
      position_target_m: 小车目标位置 [m]，对应直立目标角为 0。
      angle_limit_deg: 外环给定倾角的绝对值上限 [deg]，须在 (0,90) 内。
      force_limit_N: 最终控制力绝对值上限 [N]。
      position_pid: kp [rad/m], ki [rad/(m*s)], kd [rad*s/m],
                    integral_limit [rad]，即积分项的最大角度贡献。
      angle_pid: kp [N/rad], ki [N/(rad*s)], kd [N*s/rad],
                 integral_limit [N]，即积分项的最大力贡献。
      所有增益与积分项上限均非负；ki=0 表示关闭积分。

    外环：e_p=p_target-p，theta_ref=clip(kp*e_p + I_p - kd*v)。
    内环：e_theta=theta-theta_ref，u=clip(kp*e_theta + I_theta + kd*omega)。
    内环有意使用“实测角度减目标角度”：正力会产生负的摆角加速度。
    两环同周期执行，快慢差别由增益决定。t=0 不累积积分，此后每次累积 Ts。
    两环都有积分项限幅与条件积分；内环力饱和时还冻结外环积分。
    D 项用实测 v/omega 避免目标变化引起微分冲击；当前模型无测量噪声。
    仅用于直立附近的平衡/定位，不包含摆起控制。
    """
    _require_keys(settings, ('position_target_m', 'angle_limit_deg', 'force_limit_N',
                             'position_pid', 'angle_pid'), 'control.settings')
    period = _number(sample_period, 'sample_period', strictly_positive=True)
    target = _number(settings['position_target_m'], 'position_target_m')
    angle_limit = _number(settings['angle_limit_deg'], 'angle_limit_deg', strictly_positive=True)
    if angle_limit >= 90:
        raise ValueError('angle_limit_deg must be less than 90 degrees.')
    force_limit = _number(settings['force_limit_N'], 'force_limit_N', strictly_positive=True)
    position = _PIDLoop(settings['position_pid'], float(np.deg2rad(angle_limit)), 'position_pid')
    angle = _PIDLoop(settings['angle_pid'], force_limit, 'angle_pid')
    first_sample = True

    def compute_force(t, state):
        nonlocal first_sample
        p, v, theta, omega = state
        dt = 0.0 if first_sample else period
        theta_ref, position_i, _ = position.preview(target - p, -v, dt)
        u, angle_i, force_saturated = angle.preview(theta - theta_ref, omega, dt)
        if force_saturated:
            # 外环还需感知内环限幅；冻结外环 I 后重新计算本次命令。
            # preview 不修改状态，因此重算不会把内环积分累加两次。
            theta_ref, position_i, _ = position.preview(target - p, -v, 0.0)
            u, angle_i, _ = angle.preview(theta - theta_ref, omega, dt)
        position.integral = position_i
        angle.integral = angle_i
        first_sample = False
        return u

    return compute_force


def create_lqr_controller(parameters, settings, sample_period):
    """零阶保持离散模型上的 LQR；目标状态为 [0,0,0,0]，输出力不限幅。

    settings: Q_diag 为 [p,v,theta,omega] 对应的四个状态权重，
    默认 [10,1,100,1]；R 为控制力权重，默认 1.0。
    """
    # 1. 获得直立平衡点附近的连续线性模型
    A, B, C, D = linear_state_space(parameters)

    # 2. 零阶保持离散化
    Ad, Bd, _, _, _ = cont2discrete(
        (A, B, C, D),
        sample_period,
        method="zoh"
    )

    # 3. LQR 权重
    Q = np.diag(settings.get("Q_diag", [10, 1, 100, 1]))
    R = np.array([[settings.get("R", 1.0)]])

    # 4. 求解离散 Riccati 方程
    P = solve_discrete_are(Ad, Bd, Q, R)

    # 5. 计算最优反馈增益
    K = np.linalg.solve(
        R + Bd.T @ P @ Bd,
        Bd.T @ P @ Ad
    )

    # 6. 检查离散闭环局部稳定性
    eigenvalues = np.linalg.eigvals(Ad - Bd @ K)
    if np.any(np.abs(eigenvalues) >= 1.0):
        raise ValueError("LQR closed-loop is not stable")

    # 7. 每个采样时刻计算控制力
    def compute_force(t, state):
        x = np.asarray(state, dtype=float)
        u = -K @ x
        return float(u.item())

    return compute_force



def create_mpc_controller(parameters, settings, sample_period):
    """MPC 空壳；可用参数见模块说明，预测模型、时域与约束待设计。"""
    # 后续在此利用 parameters、Ts、settings 初始化预测模型和优化问题。
    # 尚未确定采用线性 MPC 还是 NMPC；此处不引入优化器依赖。
    def compute_force(t, state):
        p, v, theta, omega = state
        # TODO：按当前状态求解优化问题，并返回最优控制序列的首个力。
        return 0.0

    return compute_force
