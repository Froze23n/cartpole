"""离散反馈控制器：串级 PID、离散 LQR，以及 MPC 占位入口。

工厂接收模型 parameters、配置 settings 和采样周期 sample_period。
返回 controller(t, state)，state 顺序为 [p, v, theta, omega]。
控制力在两次采样之间保持不变；每次仿真创建新的控制器。
"""

import numpy as np
from scipy.signal import cont2discrete
from scipy.linalg import solve_discrete_are

from cartpole_model import linear_state_space


def tune_angle_pid(parameters, settings):
    """匹配完整五阶闭环的目标多项式；倾角环采用 PD。"""
    M, m = parameters.cart_mass, parameters.pole_mass
    L, g = parameters.length, parameters.gravity
    poles = np.asarray(settings['tuning_poles'], dtype=float)
    if poles.shape != (5,) or not np.all(np.isfinite(poles)) or np.any(poles >= 0):
        raise ValueError('tuning_poles 必须是 5 个有限负实数。')
    _, c4, c3, c2, c1, c0 = np.poly(poles)

    return {
        **settings['angle_pid'],
        'kp': (M + m) * g + M * L * (c3 + L / g * c1),
        'ki': 0.0,
        'kd': M * L * (c4 + L / g * c2 + (L / g)**2 * c0),
    }


def tune_position_pid(parameters, settings, angle_pid, sample_period):
    """与 tune_angle_pid 配套整定位置 PID，并检查完整离散闭环。"""
    M, L, g = parameters.cart_mass, parameters.length, parameters.gravity
    c2, c1, c0 = np.poly(settings['tuning_poles'])[3:]
    ka, da = angle_pid['kp'], angle_pid['kd']
    if angle_pid['ki'] != 0:
        raise ValueError('这套整定公式要求倾角环 ki=0，请配套调用 tune_angle_pid。')
    position_pid = {
        **settings['position_pid'],
        'kp': M * L * c1 / (ka * g),
        'ki': M * L * c0 / (ka * g),
        'kd': M * L * (c2 + L / g * c0) / (ka * g),
    }

    # 未触发限幅时，增广状态为 [p, v, theta, omega, position_i]。
    A, B, C, D = linear_state_space(parameters)
    Ad, Bd, _, _, _ = cont2discrete((A, B, C, D), sample_period, method='zoh')
    kp, ki, kd = (position_pid[key] for key in ('kp', 'ki', 'kd'))
    force_feedback = np.array([[ka * kp, ka * kd, ka, da]])
    closed_loop = np.block([
        [Ad + Bd @ force_feedback, -ka * Bd],
        [np.array([[-ki * sample_period, 0, 0, 0]]), np.ones((1, 1))],
    ])
    if np.any(np.abs(np.linalg.eigvals(closed_loop)) >= 1):
        raise ValueError('整定后的离散闭环不稳定：请减小采样周期或放慢目标极点后重试。')
    return position_pid


def create_pid_controller(parameters, settings, sample_period):
    """位置外环给出目标倾角，倾角内环给出水平力。"""
    position_target = settings['position_target_m']
    angle_limit = np.deg2rad(settings['angle_limit_deg'])
    force_limit = settings['force_limit_N']
    # 基准增益：需要对照时启用这两行，并注释下面的整定调用。
    # position_pid = settings['position_pid']
    # angle_pid = settings['angle_pid']

    # 默认按目标极点配套整定两环。
    angle_pid = tune_angle_pid(parameters, settings)
    position_pid = tune_position_pid(parameters, settings, angle_pid, sample_period)

    Ts = sample_period
    position_i = 0.0
    angle_i = 0.0

    def compute_force(t, state):
        nonlocal position_i, angle_i
        p, v, theta, omega = state

        # 位置外环：位置误差 -> 目标倾角。
        position_error = position_target - p
        position_p = position_pid['kp'] * position_error
        position_d = -position_pid['kd'] * v
        theta_ref = float(np.clip(position_p + position_i + position_d,
                                  -angle_limit, angle_limit))

        # 倾角内环：正力产生负角加速度，因此误差取“实际 - 目标”。
        angle_error = theta - theta_ref
        angle_p = angle_pid['kp'] * angle_error
        # D 只考虑实际倾角变化，不考虑目标倾角变化。
        angle_d = angle_pid['kd'] * omega
        u = float(np.clip(angle_p + angle_i + angle_d, -force_limit, force_limit))

        # 输出算完后更新积分并限幅，供下一次调用使用。
        position_i += position_pid['ki'] * position_error * Ts
        position_i = float(np.clip(position_i,
                                   -position_pid['integral_limit'], position_pid['integral_limit']))
        angle_i += angle_pid['ki'] * angle_error * Ts
        angle_i = float(np.clip(angle_i,
                                -angle_pid['integral_limit'], angle_pid['integral_limit']))
        return u

    return compute_force


def create_lqr_controller(parameters, settings, sample_period):
    """零阶保持离散模型上的 LQR；目标状态为 [0,0,0,0]。

    settings: Q_diag 为 [p,v,theta,omega] 对应的四个状态权重，
    默认 [10,1,100,1]；R 为控制力权重，默认 1.0。
    force_limit_N 可选；省略时不限幅，设置时必须有限且大于零。
    """
    force_limit = settings.get('force_limit_N')
    if force_limit is not None:
        force_limit = float(force_limit)
        if not np.isfinite(force_limit) or force_limit <= 0:
            raise ValueError('force_limit_N must be finite and positive.')
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
        force = float(u.item())
        return force if force_limit is None else float(np.clip(force, -force_limit, force_limit))

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
