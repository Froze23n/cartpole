# 倒立摆离散控制仿真

本项目用 Python 模拟“离散控制器驱动连续倒立摆”，用于比较不同控制方法。PID 采用位置与摆角的串级结构，以直立附近的线性模型辅助选参；LQR 基于线性化状态空间的离散模型设计，MPC 预留扩展入口。**所有控制器都在同一个原始非线性模型上检验。**

目前已实现仿真、数据记录、绘图、动画、串级 PID 和离散 LQR；MPC 仍为返回 **0 N** 的空壳。

## 1. 物理模型与控制设计模型

### 对象与坐标

采用水平小车、无质量刚杆和顶部点质量的经典模型。忽略摩擦、空气阻力、电机动态、轨道边界与碰撞。物理方程没有执行器饱和模型；PID 在控制器内部主动限制输出力。

| 符号 | 含义 | 单位 |
| --- | --- | --- |
| M、m | 小车质量、顶部质点质量 | kg |
| L | 支点到顶部质点的距离 | m |
| g | 重力加速度 | m/s² |
| p、v | 小车位置、速度，向右为正 | m、m/s |
| θ、ω | 摆角、角速度；θ=0 竖直向上，正角向右倾斜 | rad、rad/s |
| u | 作用于小车的水平力，向右为正 | N |

状态顺序固定为 `x=[p,v,θ,ω]ᵀ`，内部使用弧度，角度连续累计。顶部坐标为 `(p+L sinθ, L cosθ)`。模型类的默认参数为 M=1、m=0.1、L=0.5、g=9.81；实际运行以 JSON 为准。PID 配置的模型参数为 M=1、m=1、L=0.5、g=9.81；这些均是仿真设定值。

### 连续非线性方程

令 `d=M+m sin²θ`，有

$$
\dot p=v,\qquad
\dot v=\frac{u+mL\omega^2\sin\theta-mg\sin\theta\cos\theta}{d},
$$

$$
\dot\theta=\omega,\qquad
\dot\omega=\frac{(M+m)g\sin\theta-u\cos\theta-mL\omega^2\sin\theta\cos\theta}{Ld}.
$$

直立静止时，正向推力使小车向右加速、摆杆向左倾斜；直立平衡点在无控制时不稳定。

<details>
<summary>拉格朗日建模与能量检验</summary>

动能、势能和拉格朗日函数为

$$
T=\tfrac12(M+m)v^2+mLv\omega\cos\theta+\tfrac12mL^2\omega^2,
\qquad V=mgL\cos\theta,\qquad\mathcal L=T-V.
$$

取广义坐标 `[p,θ]` 和广义力 `[u,0]`，代入拉格朗日方程得到

$$
(M+m)\ddot p+mL\cos\theta\ddot\theta-mL\sin\theta\omega^2=u,
\qquad mL\cos\theta\ddot p+mL^2\ddot\theta-mgL\sin\theta=0.
$$

联立求解即得上述非线性方程。机械能 `E=T+V` 满足 `dE/dt=u·v`，因此 `E(t)-E(0)-∫u·v dt` 应接近零，可用于检查数值仿真的物理一致性。

</details>

### 线性化、传递函数与离散化

在静止直立平衡点附近，以状态和输入的偏差表示：

$$
\delta\dot x=A\delta x+B\delta u,\qquad
A=\begin{bmatrix}0&1&0&0\\0&0&-mg/M&0\\0&0&0&1\\0&0&(M+m)g/(ML)&0\end{bmatrix},\qquad
B=\begin{bmatrix}0\\1/M\\0\\-1/(ML)\end{bmatrix}.
$$

若输出为小车位置和摆角，则 `y=[p,θ]ᵀ`，

$$
C=\begin{bmatrix}1&0&0&0\\0&0&1&0\end{bmatrix},\qquad D=\begin{bmatrix}0\\0\end{bmatrix}.
$$

对应零初始条件下的传递函数为

$$
\frac{\Theta(s)}{U(s)}=-\frac1{MLs^2-(M+m)g},\qquad
\frac{P(s)}{U(s)}=\frac{Ls^2-g}{s^2[MLs^2-(M+m)g]}.
$$

角度通道的二阶传递函数不包含小车绝对位置和匀速平移模态，不能替代完整的四阶模型。

对于采样周期 Ts、周期内保持不变的输入，线性模型的零阶保持离散化为

$$
\delta x_{k+1}=A_d\delta x_k+B_d\delta u_k,\qquad
A_d=e^{AT_s},\qquad B_d=\int_0^{T_s}e^{A\tau}B\,d\tau.
$$

这些矩阵用于控制器设计与分析。仿真对象仍通过连续非线性方程推进，反馈接口当前可准确读取全部四个状态，不含传感器噪声和计算延迟。

### 串级 PID

位置外环给出临时目标倾角，倾角内环给出水平力。以固定位置目标 r 表示，未限幅的控制量为

$$
e_p=r-p,\qquad \theta_{\mathrm{ref}}=k_{p,p}e_p+I_p-k_{d,p}v,
$$

$$
e_\theta=\theta-\theta_{\mathrm{ref}},\qquad u=k_{p,\theta}e_\theta+I_\theta+k_{d,\theta}\omega.
$$

内环使用“实际角度减目标角度”，与本项目正力产生负角加速度的坐标约定一致：摆杆右倾时，先向右推小车接住它。D 使用实测速度，不对目标值差分，避免目标变化引起微分冲击；当前测量无噪声，未另加微分滤波器。两环按同一 Ts 更新，内环应比外环响应更快；设计外环时必须包含闭合内环后的完整动力学，不能把倾角到位置的关系简单替换成恒定比例。

I 项采用后向矩形积分 `I_k=I_{k-1}+Ki*Ts*e_k`，t=0 不积分；ki=0 时关闭该环积分。控制器限制目标倾角、输出力及 I 项贡献；输出饱和且积分会继续推向饱和时暂停积分，反向卸载仍被允许，内环力饱和时还冻结外环积分。这采用[条件积分防饱和方法](https://www.mathworks.com/help/simulink/slref/anti-windup-control-using-a-pid-controller.html)。当前配置使用位置 PID、倾角 PD；局部稳定性需检查包含四个物理状态和启用积分状态的完整离散闭环，实际效果需在非线性对象上验证。

## 2. 环境与运行

在项目根目录（包含 `run_simulation.py` 的目录）打开终端，激活环境后选择配置运行：

```bash
conda activate cartpole
python run_simulation.py --config configs/pid.json
python run_simulation.py --config configs/lqr.json
python run_simulation.py --config configs/mpc.json
```

每条命令分别执行一次仿真。省略 `--config` 时默认使用 `configs/pid.json`。编辑器中选择 `cartpole` 环境的 Python 解释器。

首次使用、尚未创建环境时执行：

```bash
conda create -n cartpole --override-channels -c conda-forge --strict-channel-priority python=3.12 numpy scipy control matplotlib sympy jupyterlab ipykernel ipympl pillow ffmpeg
```

NumPy/SciPy 用于数值计算，python-control/SymPy 用于控制分析与推导，Matplotlib 用于绘图，Pillow/FFmpeg 用于动画。需要笔记本时，在该环境运行 `jupyter lab`。

## 3. 四种时间参数：控制、求解、记录、播放

**控制器何时更新力、求解器如何计算运动、何时保存状态、视频如何播放，是四件不同的事。**

| 参数 | 作用 | 调整后的影响 |
| --- | --- | --- |
| `control.sample_period_s` | 反馈采样周期 Ts，每隔 Ts 读取当前状态并计算一次力 | 改变离散控制行为，可能影响稳定性和控制效果 |
| `simulation.max_step_s` | 连续 ODE 求解器允许的最大内部步长 | 影响数值求解精度和计算耗时；实际步长自适应，通常并不固定 |
| `simulation.output_dt_s` | 状态记录间隔，决定 `trajectory.csv` 的时间网格 | 改变记录密度、曲线细节和动画插值依据；不改变反馈更新或积分轨迹 |
| `output.animation.fps` | 每秒视频播放的帧数 | 改变动画流畅度和渲染开销；不改变物理仿真或反馈频率 |

例如：Ts=0.02 s、最大内部步长=0.01 s、状态记录间隔=0.01 s、视频帧率=20 fps、播放倍速=1 时：

- **反馈**在 0、0.02、0.04…秒执行，相当于 50 Hz；每次输出的力保持到下一次更新。
- **求解**在每个保持区间内自适应积分，单步不超过 0.01 s，也不会跨过下一次控制更新点。积分内部评估不会调用控制器。
- **记录**在 0、0.01、0.02…秒保存状态，即按 100 Hz 的时间网格记录状态；记录时刻的状态由求解器连续插值获得。
- **视频**每隔 0.05 个仿真秒取一帧。动画根据已记录的状态插值生成，不重新计算动力学。

四个参数不需要设置成相同值，也不要求周期成整数倍。更密的记录或更多视频帧不能代替更精确的数值求解；记录过稀时，提高帧率也无法补回丢失的运动细节。求解器的 `rtol`、`atol` 分别控制相对和绝对误差容限，与最大步长共同决定求解设置。

`output.animation.speed` 是播放倍速：2 表示一秒视频展示两秒仿真运动，相邻画面对应的仿真时间间隔为 `speed/fps`。仿真时长由 `simulation.duration_s` 决定，计算耗时由机器与设置决定，二者不等于视频播放时长。

对于 30 s 仿真、Ts=0.02 s、记录间隔=0.01 s：反馈更新 **1500 次**，状态记录 **3001 条**。状态包含起点和终点；终点不再更新控制力，最后一个控制区间必要时缩短到终点。

## 4. 配置与控制器接口

所有运行参数均写在 JSON 中。除上节的时间参数外，主要字段如下：

| 配置项 | 含义 |
| --- | --- |
| `parameters.M / m / L / g` | 模型参数，单位见模型表，均为有限正数 |
| `initial_state.position_m / velocity_m_s` | 初始小车位置与速度 |
| `initial_state.angle_deg / angular_velocity_deg_s` | 初始摆角与角速度；也可分别使用 `angle_rad / angular_velocity_rad_s`，同一量只选一种单位 |
| `control.file` | 反馈 Python 模块路径，相对于 JSON 所在目录解析 |
| `control.factory` | 创建控制器的工厂函数名 |
| `control.settings` | PID、LQR 参数见下文；MPC 空壳使用空对象 `{}` |
| `output.plot` | 是否生成曲线图，取 `true` 或 `false` |
| `output.animation.format` | `none` 关闭动画，或选择 `mp4`、`gif` |

三个控制器都位于 `feedback.py`：

| 配置 | 工厂函数 |
| --- | --- |
| `configs/pid.json` | `create_pid_controller` |
| `configs/lqr.json` | `create_lqr_controller` |
| `configs/mpc.json` | `create_mpc_controller` |

平台每次仿真调用一次工厂函数 `factory(parameters, settings, sample_period)`，获得 `controller(t, state)`。每个采样时刻传入当前仿真时间和状态副本 `[p,v,theta,omega]`，返回有限实数力 u，单位 N。工厂作用域可保存积分误差等内部状态，各次仿真独立初始化。可用参数和待实现位置见 `feedback.py` 注释。

`configs/template.json` 是可修改的配置模板。需要在 Python 中调试时，可调用 `simulate_feedback(...)` 完成整段仿真，或用 `CartPoleSimulator.step(u, dt)` 逐段施加由外部反馈逻辑算出的力。

### PID 参数

全部可调参数在 `control.settings` 中，均须显式填写：

| 字段 | 当前值 | 含义与单位 |
| --- | --- | --- |
| `position_target_m` | 0 | 目标小车位置，m；最终目标摆角为 0 |
| `angle_limit_deg` | 10 | 外环目标倾角绝对值上限，deg；不会直接限制真实摆角 |
| `force_limit_N` | 20 | 控制力绝对值上限，N |
| `position_pid.kp / ki / kd` | 0.16 / 0.06 / 0.20 | 外环增益，单位依次为 rad/m、rad/(m·s)、rad·s/m |
| `position_pid.integral_limit` | 0.05 | 外环 I 项的最大角度贡献，rad |
| `angle_pid.kp / ki / kd` | 100 / 0 / 20 | 内环增益，单位依次为 N/rad、N/(rad·s)、N·s/rad |
| `angle_pid.integral_limit` | 5 | 内环 I 项的最大力贡献，N；ki=0 时不起作用 |

这组参数使用**外环 PID、内环 PD**。内环积分功能已实现，但本例先关闭它，用位置外环消除位置偏差。所有增益非负，角度计算均使用弧度，只有 `angle_limit_deg` 在读取时转换成弧度。

这组增益的已验证基准条件为 M=1 kg、m=1 kg、L=0.5 m、Ts=0.02 s、初始位置 0 m、初始倾角 5°、初始速度为零，仿真 10 s。基准初态不代表可编辑的 `configs/pid.json` 当前取值，实际运行前请检查其中的 `initial_state`。参数经离散线性闭环极点检查和完整非线性仿真验证，可用于直立附近的平衡与定位；不包含摆起控制。改变质量、杆长、采样周期或增大初始偏差后，需要重新验证。基于传递函数的系统性整定分析仍可在此基础上继续开展。比较不同控制方法时，应统一模型、初态、时长、Ts 和控制力限制。

### LQR 参数

LQR 将直立处的线性状态空间模型按 Ts 作零阶保持离散化，求解离散 Riccati 方程，使用 `u=-Kx`。目标为位置、速度、摆角和角速度均为零；每次创建控制器时依据模型参数、采样周期和权重重新计算增益。

`control.settings` 可设置 `Q_diag`（按 `[p,v,theta,omega]` 顺序的四个非负状态权重）和 `R`（正的控制力权重）。省略时分别使用 `[10,1,100,1]` 和 `1.0`，因此当前 `configs/lqr.json` 中的 `{}` 会启用默认权重。调参时可显式写为：

```json
"settings": {"Q_diag": [10, 1, 100, 1], "R": 1.0}
```

当前 LQR 不限制输出力，也不包含摆起控制。当前 PID 与 LQR 配置的模型质量和初态等条件不同，且 PID 有力限幅，尚不能直接用两份默认运行结果作公平比较。

## 5. 输出与结果检查

每次运行新建 `results/YYYYMMDD-HHMMSS-微秒/`，目录名使用仿真开始时的 Asia/Shanghai 时间。终端结束信息会显示结果目录。

| 文件 | 内容 |
| --- | --- |
| `config.json`、`feedback_source.py` | 本次配置与反馈模块快照 |
| `trajectory.csv` | 时间、状态、保持力、加速度、角度的度数版本、能量、外力做功与能量残差 |
| `control.csv` | 每次反馈时刻的状态和控制力，按反馈采样周期记录 |
| `metadata.json` | 运行状态、时间、实际参数、求解设置、软件版本与结果摘要 |
| `response.png` | 状态、力和能量曲线，开启绘图时生成 |
| `animation.mp4` 或 `animation.gif` | 开启动画时生成 |

先查看 `response.png`，再根据需要读取 CSV 或播放动画。MPC 空壳的控制力应全为零；非零初始偏角下不会主动保持直立。PID 和 LQR 应按实际状态和参数产生反馈，需同时检查摆角、位置、速度及控制力。零力时机械能应近似守恒，非零力时可检查能量变化与输入功是否一致。

按当前格式保存的归档配置可用于重跑：

```bash
python run_simulation.py --config "results/<运行目录名>/config.json"
```

将 `<运行目录名>` 替换成实际目录名。程序加载归档中的反馈模块快照，并为新运行创建独立目录。自动快照仅包含指定模块；若控制器依赖其他自编文件，也需保留这些依赖。

## 6. 自动测试

在项目根目录、已激活 `cartpole` 环境的终端运行：

```bash
python -m unittest discover -s tests -v
```

`unittest` 是 **Python 标准库中的测试模块**，无需在项目中创建同名文件。`python -m unittest` 启动该模块，`discover` 自动查找测试，`-s tests` 指定从 `tests/` 目录查找 `test*.py` 文件，`-v` 逐项显示结果。

本项目的测试位于 `tests/test_model.py`、`test_simulation.py`、`test_feedback.py`、`test_archive.py`、`test_pid.py` 和 `test_lqr.py`，检查动力学、能量与动量、反馈时序、配置、归档重跑，以及 PID 的方向、记忆、限幅和指定工况下的非线性响应。LQR 测试检查反馈增益、离散闭环极点、参数变化及非线性对象上的平衡效果。PID 收敛测试读取 `configs/pid.json` 的模型参数、控制参数和采样周期，但初态、目标位置与 10 s 时长由测试用例指定，不直接使用 JSON 的 `initial_state` 和 `duration_s`。运行结束显示 `OK` 表示这些测试通过，不等于当前 JSON 初态已经通过仿真验收，也不能由此推断任意配置都能稳定倒立摆。

## 7. 文件导航

| 文件或目录 | 职责 |
| --- | --- |
| `cartpole_model.py` | 非线性动力学、几何、能量、线性状态空间与传递函数 |
| `cartpole_simulation.py` | 积分、反馈时序、单步调试与数据导出 |
| `feedback.py`、`configs/` | 控制器实现入口与运行配置 |
| `run_simulation.py` | 加载配置、组织仿真和结果归档 |
| `cartpole_visualization.py` | 绘图和动画 |
| `tests/` | 自动测试 |
| [作业要求.pdf](作业要求.pdf) | 课程要求原件 |
| [AGENTS.md](AGENTS.md) | AI 协作约定与工作交接 |
