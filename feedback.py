"""离散反馈控制器占位：PID、LQR、MPC 当前均返回 0 N，尚未实现算法。

JSON 的 control.factory 选择下面的工厂函数，每次仿真创建全新实例。
三个工厂都可使用以下参数：
- parameters.cart_mass：底座质量 M [kg]。
- parameters.pole_mass：顶部质点质量 m [kg]。
- parameters.length：杆长 L [m]，支点至质点的距离。
- parameters.gravity：重力加速度 g [m/s²]。
- settings：来自 JSON 的 control.settings；目前为空，尚无生效的算法参数。
- sample_period：离散控制采样周期 Ts [s]。

返回的控制函数在每个采样时刻被调用一次，区间内保持输出力：
- t：当前仿真时间 [s]。
- state：当前状态副本 [p, v, theta, omega]，单位 [m, m/s, rad, rad/s]。
  p 向右为正；theta=0 竖直向上，正角向右倾斜；角度连续累计。
- 返回值：施加在小车上的有限实数力 u [N]，向右为正。
可在工厂作用域保存控制器内部状态，每次仿真独立初始化。
"""


def create_pid_controller(parameters, settings, sample_period):
    """PID 空壳；可用参数见模块说明，增益与控制结构待设计。"""
    # 后续在此读取 settings 中的增益，并初始化积分、上一时刻误差等状态。
    def compute_force(t, state):
        p, v, theta, omega = state
        # TODO：根据当前状态和 Ts 计算离散 PID 控制力。
        return 0.0

    return compute_force


def create_lqr_controller(parameters, settings, sample_period):
    """LQR 空壳；可用参数见模块说明，权重与反馈增益待设计。"""
    # 后续在此由 parameters 构造线性模型，结合 Ts 和 settings 设计反馈增益。
    def compute_force(t, state):
        p, v, theta, omega = state
        # TODO：根据状态偏差计算 LQR 控制力。
        return 0.0

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
