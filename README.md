# 倒立摆仿真

同一个连续非线性倒立摆，由离散反馈接口每隔 Ts 秒计算一次力，并在下一周期前保持该力。当前 `feedback.py` 提供 PID、LQR、MPC 三个空壳，均返回 0 N，尚未实现控制算法；基于传递函数的 PID、基于线性状态空间的 LQR 待设计，MPC 先预留入口。

## 运行与环境

```bash
conda activate cartpole
cd /Users/zz/Desktop/invert
python run_simulation.py
```

默认读取 [configs/pid.json](configs/pid.json)。可用 `python run_simulation.py --config configs/lqr.json` 或 `--config configs/mpc.json` 选择另外两个入口。所有可调参数在 JSON 中；输出固定为 `results/开始时间/`，时间使用 Asia/Shanghai，格式为 `YYYYMMDD-HHMMSS-微秒`，每次独立建目录。

环境已建好：Python 3.12，NumPy/SciPy 负责数值计算，python-control 负责控制分析，Matplotlib 绘图，SymPy 符号推导，JupyterLab/ipykernel/ipympl 提供笔记本，Pillow/FFmpeg 导出动画。只有重建环境时才执行：

```bash
conda create -n cartpole --override-channels -c conda-forge --strict-channel-priority python=3.12 numpy scipy control matplotlib sympy jupyterlab ipykernel ipympl pillow ffmpeg
```

编辑器选择 `/opt/miniconda3/envs/cartpole/bin/python`；笔记本启动命令是 `jupyter lab`，交互绘图可用 `%matplotlib widget`。

## 配置与反馈接口

| JSON 配置 | 含义 |
| --- | --- |
| `parameters.M / m / L / g` | 底座质量 kg、顶部质点质量 kg、杆长 m、重力加速度 m/s²，均为有限正数 |
| `initial_state` | `position_m`、`velocity_m_s`、`angle_deg`、`angular_velocity_deg_s`；也可分别使用 `angle_rad`、`angular_velocity_rad_s`，同一量不混用两种单位 |
| `simulation.duration_s / output_dt_s` | 仿真时长／状态记录间隔 |
| `simulation.max_step_s / rtol / atol` | 连续求解器最大内部步长、相对／绝对误差容限 |
| `control.sample_period_s` | 反馈采样周期 Ts |
| `control.file / factory / settings` | 反馈模块路径、工厂函数名和控制器参数 |
| `output.plot` | 是否输出曲线 |
| `output.animation.format / fps / speed` | `none`、`mp4` 或 `gif`，帧率、播放倍速 |

输入文件路径相对于 JSON 所在目录。`output` 仅支持 `plot` 和 `animation`；旧版 `directory`、`timestamp_subdirectory` 字段已移除，使用时会报错。

平台每次运行调用配置中 `control.factory` 指定的工厂函数创建新实例，再于每个采样时刻调用 `controller(t, state) -> u`。`state=[p,v,theta,omega]` 是当前状态的副本，单位为 m、m/s、rad、rad/s；返回值是有限实数，单位 N。控制器可用闭包或类实例保存积分等内部状态，各次运行互不继承记忆。

| 配置 | feedback.py 中的工厂函数 |
| --- | --- |
| `configs/pid.json` | `create_pid_controller(parameters, settings, sample_period)` |
| `configs/lqr.json` | `create_lqr_controller(parameters, settings, sample_period)` |
| `configs/mpc.json` | `create_mpc_controller(parameters, settings, sample_period)` |

三个入口的 `settings` 暂为空，参数含义及后续实现位置已注释在 `feedback.py` 中。原速度阻尼示例与 `configs/simulation.json` 已移除。现有 `configs/template.json` 也已改接 PID 空壳，保留其初始位置 0.3 m、直立和 MP4 输出设置。

**采样周期、记录间隔、求解器内部步长彼此独立。** 默认 30 s、Ts=0.02 s、记录间隔 0.01 s：反馈计算 1500 次，输出 3001 条状态；终点不再计算新力，最后不足一个周期时缩短保持区间。求解器内部评估不会重复调用控制器。仿真按逻辑时间运行，不等待实际时间流逝。

Python 调试接口：`simulate_feedback(controller=..., sample_period=..., params=..., initial_state=..., duration=..., output_dt=...)` 返回结果；`CartPoleSimulator.step(u, dt)` 可单步推进。接口实现在 `cartpole_simulation.py`。

仅支持 `control` 反馈函数配置；旧版 `force` 配置及 `simulate(force=...)` 接口已移除。需要零力物理检查时，让反馈函数返回 `0.0` 即可。

## 结果与重跑

每次开始运行就归档输入，成功后在同一目录保存全部结果：

| 文件 | 内容 |
| --- | --- |
| `config.json` | 本次使用的 JSON 原样副本；不再嵌入 metadata 重复保存 |
| `feedback_source.py` | 本次反馈模块快照 |
| `trajectory.csv` | 时间、四个状态、保持力、加速度、角度的度数版本、能量、外力做功及能量残差 |
| `control.csv` | 每次反馈时刻、当时的四个状态和输出力；独立于状态记录频率 |
| `metadata.json` | 开始／结束时间、状态、实际使用的参数与求解设置、版本、摘要和归档文件索引 |
| `response.png`、`animation.mp4/gif` | 按配置生成的曲线与动画 |

CSV 使用 17 位有效数字保留浮点精度，不再另存重复的 NPZ；完整力序列只存于相应 CSV，不再复制进 metadata。失败运行也保留配置和失败原因。反馈模块依赖的其他自编文件需一并维护，本平台自动快照的是指定模块。

更改当前配置不会影响已完成的记录。符合当前格式的归档配置可直接重跑，程序会使用该目录的反馈快照，并创建一个新的时间目录：

```bash
python run_simulation.py --config results/某次开始时间/config.json
```

历史结果均保留：已有时间戳的目录已移至 results 根下；旧版未记录真实开始时间的案例放在 `results/legacy/`，其配置从旧 metadata 恢复并标明来源，不补造开始时间。原 examples 中不同案例的参数随对应结果归档。旧版开环结果仅作历史记录，其 `force` 配置不能再由当前程序重跑。含旧输出字段的反馈配置也不再直接支持；需要重跑时另建配置，删除旧字段并将 `control.file` 指向原反馈快照，不修改原始归档。

## 验收

```bash
python -m unittest discover -s tests -v
```

测试覆盖动力学、能量／动量、离散力保持、反馈调用时序、状态记忆重置、输入快照及归档重跑。运行默认配置后，应有 1500 条控制记录、3001 条状态记录；三个占位入口的 `control.csv` 力应全为 0；这只验收调用与输出链路，不表示已实现控制。将动画 format 改为 `mp4` 即可生成动画。

零初始状态、零力应保持平衡；初始右倾 5°、零力时会向右倒，且机械能近似守恒。受力时检查 `E(t)-E(0)-∫u·v dt` 的数值残差。当前零力空壳不会稳定倒立摆；角度连续累计，350° 与 −10° 几何姿态相同。

## 模型与后续设计公式

采用平面小车、无质量刚杆和顶部点质量；没有摩擦、空气阻力、碰撞、轨道边界、推力饱和或电机动态。所有方法都检验于同一原始非线性对象。p、u 向右为正，θ=0 竖直向上，正角度向右倾斜，竖直坐标向上。反馈默认能准确读取全部四个状态，尚无传感器噪声、估计或计算延迟。

<details>
<summary>建模推导、线性化与传递函数（展开）</summary>

顶部坐标与速度：

\[
x_m=p+L\sin\theta,\quad y_m=L\cos\theta,\quad
\dot x_m=v+L\omega\cos\theta,\quad\dot y_m=-L\omega\sin\theta.
\]

动能、势能与拉格朗日函数：

\[
T=\tfrac12(M+m)v^2+mLv\omega\cos\theta+\tfrac12mL^2\omega^2,\quad
V=mgL\cos\theta,\quad\mathcal L=T-V.
\]

对广义坐标 q=[p,θ]，广义力为 [u,0]；将其代入 `d/dt(∂ℒ/∂q̇)−∂ℒ/∂q=Q`，得到

\[
(M+m)\ddot p+mL\cos\theta\ddot\theta-mL\sin\theta\omega^2=u,\qquad
mL\cos\theta\ddot p+mL^2\ddot\theta-mgL\sin\theta=0.
\]

质量矩阵行列式为 `mL²(M+m sin²θ)>0`。令 d=M+m sin²θ，完整非线性状态方程为

\[
\dot p=v,\quad
\dot v=\frac{u+mL\omega^2\sin\theta-mg\sin\theta\cos\theta}{d},\quad
\dot\theta=\omega,\quad
\dot\omega=\frac{(M+m)g\sin\theta-u\cos\theta-mL\omega^2\sin\theta\cos\theta}{Ld}.
\]

机械能 E=T+V 满足 `dE/dt=u·v`。系统有两个机械自由度、四个一阶状态、一个力输入。任意固定 p、其余状态及 u 为 0 都是直立平衡点；θ=0、静止时正力使小车向右加速、杆向左倾斜。

在直立平衡点只保留扰动的一阶项（包含舍弃 ω²sinθ 等高阶项），得到 `δẋ=Aδx+Bδu`，x=[p,v,θ,ω]ᵀ：

\[
A=\begin{bmatrix}0&1&0&0\\0&0&-mg/M&0\\0&0&0&1\\0&0&(M+m)g/(ML)&0\end{bmatrix},\quad
B=\begin{bmatrix}0\\1/M\\0\\-1/(ML)\end{bmatrix},\quad
C=\begin{bmatrix}1&0&0&0\\0&0&1&0\end{bmatrix},\quad D=\begin{bmatrix}0\\0\end{bmatrix}.
\]

输出 y=[p,θ]ᵀ 不等同于已测得速度。零初始条件下的线性传递函数为

\[
\frac{\Theta(s)}{U(s)}=-\frac1{MLs^2-(M+m)g},\qquad
\frac{P(s)}{U(s)}=\frac{Ls^2-g}{s^2[MLs^2-(M+m)g]}.
\]

角度通道隐藏了小车绝对位置与匀速平移，不应把其二阶分母看成完整系统的阶数。非零初始条件由状态空间或原始方程处理。

默认 M=1 kg、m=0.1 kg、L=0.5 m、g=9.81 m/s² 是选定的仿真值，并非实测：A 的两个非零重力系数为 −0.981、21.582，B=[0,1,0,−2]ᵀ；极点为 0、0、±4.645643，直立开环不稳定。完整线性模型的能控秩及上述输出的能观秩均为 4。此时 `Θ/U=−2/(s²−21.582)`、`P/U=(s²−19.62)/[s²(s²−21.582)]`；5°、静止、零力时初始加速度约为 −0.085110 m/s²、1.879567 rad/s²。

线性设计模型的零阶保持离散化为

\[
x_{k+1}=A_dx_k+B_du_k,\quad A_d=e^{AT_s},\quad B_d=\int_0^{T_s}e^{A\tau}B\,d\tau.
\]

可使用 `scipy.signal.cont2discrete(..., method='zoh')`。A 不可逆，不能使用要求 A 可逆的公式；这些离散矩阵也不是原始非线性对象的全局精确表达。控制效果始终通过积分原始方程检验。

参考：[MIT Cart-Pole 模型](https://underactuated.csail.mit.edu/acrobot.html)。该资料以下垂为零，与这里的约定满足 `θ_MIT=π−θ`，不能直接混用矩阵符号。

</details>

代码分工：`cartpole_model.py` 物理方程，`cartpole_simulation.py` 积分与反馈时序，`feedback.py` 控制接口，`run_simulation.py` 配置与归档，`cartpole_visualization.py` 可视化。`tests/` 保留自动检查；根目录的 [作业要求.pdf](作业要求.pdf) 是课程要求原件，[AGENTS.md](AGENTS.md) 记录 AI 协作约定与最新交接状态。
