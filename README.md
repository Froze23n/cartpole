# 倒立摆离散控制仿真

本项目用 Python 模拟“离散控制器驱动连续倒立摆”，用于比较不同控制方法。PID 采用位置与摆角的串级结构，以直立附近的线性模型辅助选参；LQR 基于线性化状态空间的离散模型设计，MPC 预留扩展入口。**所有控制器都在同一个原始非线性模型上检验。**

目前已实现仿真、数据记录、绘图、动画、串级 PID 和离散 LQR；MPC 仍为返回 **0 N** 的空壳。

## 1. 物理模型与控制设计模型

### 对象与坐标

采用水平小车、无质量刚杆和顶部点质量的经典模型。忽略摩擦、空气阻力、电机动态、轨道边界与碰撞。物理方程没有执行器饱和模型；PID 和配置了力上限的 LQR 在控制器内部限制输出力。

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

位置外环给出临时目标倾角，倾角内环给出水平力。`feedback.py` 的 `compute_force` 将两环的误差、P、I、D 和限幅依次展开。以固定位置目标 r 表示，第 k 次采样时：

$$
e_{p,k}=r-p_k,\qquad
\theta_{\mathrm{ref},k}=\operatorname{sat}_{\theta_{\max}}\left(
 k_{p,p}e_{p,k}+I_{p,k}-k_{d,p}v_k\right),
$$

$$
e_{\theta,k}=\theta_k-\theta_{\mathrm{ref},k},\qquad
u_k=\operatorname{sat}_{u_{\max}}\left(
 k_{p,\theta}e_{\theta,k}+I_{\theta,k}+k_{d,\theta}\omega_k\right).
$$

这里 `sat_a(z)` 表示将 z 限制在 `[-a,a]`。先使用现有积分项计算本次输出，再按本次误差更新积分，供下一次调用使用：

$$
I_0=0,\qquad I_{k+1}=\operatorname{sat}_{I_{\max}}\left(I_k+k_iT_se_k\right).
$$

I 存储已乘过 ki 的积分贡献，`integral_limit` 限制该贡献。第一次输出自然使用零积分，无需首拍标志；本次误差只影响下一拍及之后的积分项。这对应前向矩形积分，ki=0 关闭对应积分。

内环使用“实际角度减目标角度”，与本项目正力产生负角加速度的坐标约定一致。**外环 D 直接使用 -v，内环 D 直接使用 omega**；首次采样也使用初始速度。固定位置目标下，位置误差变化率为 -v；内环采用对测量量的微分，不包含目标倾角变化率，因此不是完整的误差导数 `omega-d(theta_ref)/dt`。

平台在每个采样时刻提供数值积分得到的当前速度和角速度，不含传感器噪声。相邻位置/摆角的差分给出上一采样区间的平均变化率，与当前瞬时值一般不同；当前实现直接利用已有状态，未加微分滤波。控制力仍只在每隔 Ts 时更新并保持，I 仍按离散求和更新，因此仍是离散控制。

学习版保留目标倾角、输出力和积分项限幅，不进行饱和时的条件积分或两环联动冻结。积分项限幅限制了累积量，但在输出饱和时仍可能累积到上限，因此解除饱和或误差归零后仍可能保留 I 项。参数按下文约定填写，控制函数不另加通用参数校验。

两环按同一 Ts 更新，当前配置使用位置 PID、倾角 PD；内环应比外环响应更快。设计外环时必须包含闭合内环后的完整动力学；分析离散闭环时还需计入启用的积分状态，实际效果在非线性对象上验证。

### 从传递函数推导 PID 整定公式

采用**完整闭环极点配置**：先选期望的闭环极点，再比较特征多项式的系数，算出两环增益。以下推导针对直立附近、未触发限幅、固定位置目标的情况，保留现有的**位置 PID＋倾角 PD**，即内环 `ki=0`。不是两环各自独立套用一组经验公式。

记外环增益为 `Kp、Ki、Kd`，内环增益为 `ka、da`，分别对应 `angle_pid.kp、angle_pid.kd`。内环使用测量微分：

$$
u=k_a(\theta-\theta_{\mathrm{ref}})+d_a\dot\theta.
$$

由上面的力到摆角传递函数，闭合倾角环得到

$$
\frac{\Theta}{\Theta_{\mathrm{ref}}}
=\frac{k_a}{MLs^2+d_as+k_a-(M+m)g}.
$$

线性化摆杆方程还给出 `s²P=(g−Ls²)Θ`。因此外环面对的**完整对象**为

$$
\frac{P}{\Theta_{\mathrm{ref}}}
=\frac{k_a(g-Ls^2)}{s^2[MLs^2+d_as+k_a-(M+m)g]}.
$$

它包含小车平移的两个积分环节，以及正实零点 `sqrt(g/L)`，不能只用倾角闭环的二阶模型整定位置环。计算特征方程时将位置目标置零，此时外环等效控制器为 `Kp+Ki/s+Kd·s`；对测量速度微分与误差微分在这个固定目标下相同。闭环特征多项式为

$$
\begin{aligned}
D(s)={}&s^3[MLs^2+d_as+k_a-(M+m)g]\\
&+k_a(g-Ls^2)(K_ds^2+K_ps+K_i).
\end{aligned}
$$

四个物理状态加一个位置积分状态，共五阶。选择五个目标极点 `λ₁…λ₅`，展开目标多项式：

$$
D_\star(s)=\prod_{j=1}^{5}(s-\lambda_j)
=s^5+c_4s^4+c_3s^3+c_2s^2+c_1s+c_0.
$$

令 `D(s)=ML·D★(s)`，按同次幂比较系数，即得：

$$
\boxed{k_a=(M+m)g+ML\left(c_3+\frac{L}{g}c_1\right)},\qquad
\boxed{d_a=ML\left(c_4+\frac{L}{g}c_2+\frac{L^2}{g^2}c_0\right)},
$$

$$
\boxed{K_p=\frac{MLc_1}{k_ag}},\qquad
\boxed{K_i=\frac{MLc_0}{k_ag}},\qquad
\boxed{K_d=\frac{ML}{k_ag}\left(c_2+\frac{L}{g}c_0\right)}.
$$

`tune_angle_pid` 计算前两个增益，`tune_position_pid` 利用前者的结果计算后三个。**两函数配套使用**；五个目标极点属于完整闭环，不能将其中两个直接当成内环极点、另外三个当成外环极点。当前接口接受五个有限负实数，单位为 s⁻¹；极点绝对值增大通常意味着更快的期望响应，但也会增加控制需求和采样要求。这是一套可解释的初始整定方法，不保证最优性能或大角度摆起。

连续目标极点稳定还不够。使用实际 Ts 将原四阶线性对象作 ZOH 离散化，并按代码中“输出后更新 I”的时序，组成增广状态 `[p,v,theta,omega,I_p]` 的离散矩阵（以目标平衡点为原点）：

$$
F=\begin{bmatrix}k_aK_p&k_aK_d&k_a&d_a\end{bmatrix},\qquad
A_{\mathrm{cl},d}=
\begin{bmatrix}
A_d+B_dF&-k_aB_d\\
\begin{matrix}-K_iT_s&0&0&0\end{matrix}&1
\end{bmatrix}.
$$

整定函数要求这些离散特征值全部满足 `|z|<1`，否则报错。连续目标极点不等于实际离散极点；改变 Ts 后要重新检查。这个检查针对未饱和的局部模型，最终仍须在原始非线性对象上仿真。离散稳定性判据可参考 [MathWorks 的极点说明](https://www.mathworks.com/help/control/ref/dynamicsystem.pole.html)。

### 固定极点与测试范围

后续 PID 测试统一使用 `[-1, -1.2, -1.5, -2, -2.5]`，保持目标倾角上限 10°、力上限 20 N 及现有积分贡献上限，不按初态单独调参。选择依据是已测较大初始偏差下的恢复表现，接受小扰动响应稍慢的代价；这不是最优性结论，也不保证 60°、90°或180°恢复。仍使用完整五阶系数匹配，不增加共轭复极点或新的整定方式。

以下说明用于解释固定控制器的适用边界，不作为继续搜索参数的任务。

对于仅有初始倾角、初始角速度为零的状态，由非线性方程可得：在 `0<θ<90°` 时，要使最初的角加速度指向直立，必须满足

$$
u>(M+m)g\tan\theta.
$$

这是初始加速度条件，不是全局吸引域边界。当前 M=m=1 时，60° 对应约 33.98 N；20 N 力上限下，即使控制器要求更大的力，首拍也无法使角加速度反向。讨论摆起或后续可恢复性还需要完整轨迹，不能仅据此判定全局不可能。

| 调节量 | 需要观察的作用与代价 |
| --- | --- |
| `tuning_poles` | 同时影响恢复速度、两环增益、力需求及离散稳定性；不能只按“越快越好”选择 |
| `angle_limit_deg` | 限制外环允许的临时倾角，不限制实际摆角；太小会减弱位置恢复和制动，放宽后要检查真实倾角及小车行程 |
| `position_pid.integral_limit` | 限制位置积分贡献；过大可能延长饱和后的恢复，过小可能削弱持续扰动下的偏差补偿；无扰动初态实验不能代替抗扰测试 |
| `angle_pid.integral_limit` | 当前公式令角度 ki=0，I 从零开始，因此调整该上限不影响当前控制 |
| `force_limit_N` | 是执行器能力的实验假设；加大后恢复范围扩大不应归功于控制算法，PID/LQR 比较时须保持一致 |

极点与目标倾角上限需要配套考虑。内环采用 PD，在小角度模型中从目标倾角到实际倾角的直流增益为 `ka/[ka−(M+m)g]`，通常不等于 1。可用“目标倾角暂时保持、实际角速度与角加速度接近零”的持续加速阶段作估计（小车此时不是静止平衡）：

$$
G=(M+m)g,\qquad u=G\tan\theta,\qquad
\theta_{\mathrm{ref}}=\theta-\frac{G}{k_a}\tan\theta.
$$

应同时检查所需力是否超过力上限，以及该近直立角度分支是否仍存在。当 `ka>G` 时，上式在正角区的转折点为 `θ*=arccos(sqrt(G/ka))`；目标倾角超过该分支的最大参考值后，不能再按近直立恒角度跟随解释。降低 ka 可能放大同一目标倾角对应的实际摆动，因此“保持 angle_limit 不变”也不保证真实摆角范围不变。这是配套设计的分析线索，不替代完整离散非线性仿真。

当前只做 I 项幅值限制，不等同于跟踪执行器饱和的完整 anti-windup。条件积分或回算可以作为后续独立的结构实验；两环耦合时还要考虑内环限幅对外环积分的影响。参见 [MathWorks 的抗积分饱和说明](https://www.mathworks.com/help/simulink/slref/anti-windup-control-using-a-pid-controller.html)。

可选分析工具 `analyze_pid_tuning.py` 用 JSON 定义变体及统一初态，不属于后续固定参数的日常测试流程；三份探索定义显式保留各自极点，不随日常配置改变：

```bash
python analyze_pid_tuning.py --config configs/archive/pid_tuning_study.json
python analyze_pid_tuning.py --config configs/archive/pid_tuning_refinement.json
python analyze_pid_tuning.py --config configs/archive/pid_tuning_validation.json
```

三份实验定义分别用于单因素筛查、边界附近细查和候选参数的原八组/反向/混合初态验证。每次试验仍生成独立的时间目录，保存真实配置、反馈快照和完整轨迹；批次索引目录另存实验定义、基准配置、分析脚本快照、`summary.json` 与恢复情况图。批量试验关闭逐例曲线和视频，可用其中任一归档配置复跑。`pid_diagnostics.csv` 记录重建的积分贡献、限幅前后倾角目标和力，并核对重建力与实际控制记录一致。

本实验约定 20 s 内，最后 5 s 的全部记录均满足 `|p-r|、|v|<0.01`（SI 单位）且 `|θ|、|ω|<0.1°、0.1°/s` 才为达标。恢复时间取四量同时进入且在余下记录中持续满足阈值的时刻。图中区分按时达标、较晚进入阈值、20 s 内未进入阈值三类；后两者不能一概称为失稳。还需联合检查最大位置偏移、控制力 RMS 和各项饱和时间占比。有限初态网格是恢复能力的抽样，不是四维全局吸引域证明；本模型没有轨道边界，收敛也不意味着行程可用于实物。

## 2. 环境与运行

在项目根目录（包含 `run_simulation.py` 的目录）打开终端，激活环境后选择配置运行：

```bash
conda activate cartpole
python run_simulation.py --config configs/pid_06_angle_30deg.json
python run_simulation.py --config configs/lqr_06_angle_30deg.json
python run_simulation.py --config configs/examples/mpc.json
```

每条命令分别执行一次仿真。省略 `--config` 时默认使用 `configs/pid_06_angle_30deg.json`。编辑器中选择 `cartpole` 环境的 Python 解释器。

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
| `configs/pid_01_*.json` 至 `configs/pid_17_*.json` | `create_pid_controller` |
| `configs/lqr_01_*.json` 至 `configs/lqr_17_*.json` | `create_lqr_controller` |
| `configs/examples/mpc.json` | `create_mpc_controller` |

平台每次仿真调用一次工厂函数 `factory(parameters, settings, sample_period)`，获得 `controller(t, state)`。每个采样时刻传入当前仿真时间和状态副本 `[p,v,theta,omega]`，返回有限实数力 u，单位 N。工厂作用域可保存积分误差等内部状态，各次仿真独立初始化。可用参数和待实现位置见 `feedback.py` 注释。

`configs/examples/template.json` 是独立的可修改模板，不属于下列控制变量测试组。需要在 Python 中调试时，可调用 `simulate_feedback(...)` 完成整段仿真，或用 `CartPoleSimulator.step(u, dt)` 逐段施加由外部反馈逻辑算出的力。

### PID 参数

可调参数位于 `control.settings`。默认使用 `tuning_poles` 配套整定两环；`position_pid`、`angle_pid` 中的手填增益保留为对照基准，其中的 `integral_limit` 仍参与实际控制：

| 字段 | JSON 中的值 | 含义与单位 |
| --- | --- | --- |
| `position_target_m` | 0 | 目标小车位置，m；最终目标摆角为 0 |
| `angle_limit_deg` | 10 | 外环目标倾角绝对值上限，deg；不会直接限制真实摆角 |
| `force_limit_N` | 20 | 控制力绝对值上限，N |
| `position_pid.kp / ki / kd` | 0.16 / 0.06 / 0.20 | 外环增益，单位依次为 rad/m、rad/(m·s)、rad·s/m |
| `position_pid.integral_limit` | 0.05 | 外环 I 项的最大角度贡献，rad |
| `angle_pid.kp / ki / kd` | 100 / 0 / 20 | 内环增益，单位依次为 N/rad、N/(rad·s)、N·s/rad |
| `angle_pid.integral_limit` | 5 | 内环 I 项的最大力贡献，N；ki=0 时不起作用 |
| `tuning_poles` | `[-1, -1.2, -1.5, -2, -2.5]` | 完整连续闭环的五个目标极点，s⁻¹ |

手填基准和默认公式整定均使用**外环 PID、内环 PD**。内环积分功能已实现，但这套整定令其 ki=0，用位置外环消除位置偏差。所有增益非负，角度计算均使用弧度，只有 `angle_limit_deg` 在读取时转换成弧度。

下列初态测试统一使用同一组目标极点及限幅，不为不同案例单独调参。后续重点是大角度及较大速度、角速度偏差，5° 仅作基准检查。当前公式仍基于直立附近的模型且不包含摆起算法，必须记录大扰动下的成功范围、失败表现和饱和情况。改变极点、模型或采样周期后应重新验证；PID 与 LQR 比较还需统一模型、初态、时长、Ts 和力限制。

在 `feedback.py` 的 `create_pid_controller` 开头选择参数来源：

```python
# 基准增益：需要对照时启用这两行，并注释下面的整定调用。
# position_pid = settings['position_pid']
# angle_pid = settings['angle_pid']

# 默认按目标极点配套整定两环。
angle_pid = tune_angle_pid(parameters, settings)
position_pid = tune_position_pid(parameters, settings, angle_pid, sample_period)
```

**默认启用公式整定，JSON 的 `tuning_poles` 固定为上述统一配置。** 要比较手填基准，同时启用前两行赋值、注释后两行整定调用；两环必须配套切换。整定只在创建控制器时计算一次，读取 JSON 的物理参数与目标极点来计算 `kp/ki/kd`，保留目标、倾角上限、力上限和积分贡献上限，不回写 JSON。公式模式下修改基准 `kp/ki/kd` 不影响实际输出。

以测试组 M=1、m=1、L=0.5、g=9.81 和上述目标极点为例，公式得到：

| 环路 | kp | ki | kd |
| --- | ---: | ---: | ---: |
| 位置环 | 0.046591 | 0.013703 | 0.062439 |
| 倾角环 | 33.474817 | 0 | 5.145074 |

这一组目标极点作为后续固定测试方案，不是最优解。`configs/examples/template.json` 使用不同质量，会依据其自身模型重新计算增益。归档中的 `feedback_source.py` 会保留本次是否启用了上述两行，因此相同 JSON 也应结合控制器快照理解。

### PID/LQR 配对初态实验

正式实验为 **17 对 PID/LQR，共 34 份配置**，均仿真20 s、输出曲线和MP4（20 fps、1倍速）。统一 M=m=1、L=0.5、g=9.81、Ts=0.02 s、记录间隔与最大积分步长0.01 s、力上限20 N。PID固定极点 `[-1,-1.2,-1.5,-2,-2.5]`，各案例控制器参数不变。

| 编号 | 文件后缀（前缀为 pid 或 lqr） | 唯一非零初态 |
| --- | --- | --- |
| 01–12 | `01_angle_05deg` 至 `12_angle_60deg` | θ=5°～60°，每5°一组 |
| 13 | `13_horizontal_90deg` | θ=90° |
| 14 | `14_downward_180deg` | θ=180° |
| 15 | `15_position_0p3m` | p=0.3 m |
| 16 | `16_velocity_2mps` | v=2 m/s |
| 17 | `17_angular_velocity_180degps` | ω=180°/s |

同编号PID/LQR仅控制算法及其专属参数不同，用于比较方法；同控制器的角度组只改变初始角度，用于比较恢复范围。位置、速度、角速度案例分别与同编号另一控制器对比，不能把不同类型初态之间的差异解释为单因素效应。默认入口为 `configs/pid_06_angle_30deg.json`。

`configs/` 根目录仅放正式实验。`configs/examples/` 放独立模板、LQR示例和MPC空壳示例（也开启MP4），不纳入正式对比。`configs/archive/` 保留旧极点探索批次定义，仅供追溯，不属于本轮实验，也不直接传给 run_simulation。历史 results 不修改。

需要运行时，可选择单份配置，或遍历 `configs/pid_[0-9][0-9]_*.json`、`configs/lqr_[0-9][0-9]_*.json`。角度连续累计，不在±180°折返；目标倾角上限不限制实际摆角。

每个案例分别归档到新的时间目录；检查曲线、视频和最终状态，区分“仿真正常完成”和“控制成功”。

### LQR 参数

LQR 将直立处的线性状态空间模型按 Ts 作零阶保持离散化，求解离散 Riccati 方程，使用 `u=-Kx`。目标为位置、速度、摆角和角速度均为零；每次创建控制器时依据模型参数、采样周期和权重重新计算增益。

`control.settings` 可设置 `Q_diag`（按 `[p,v,theta,omega]` 顺序的四个非负状态权重）、`R`（正的控制力权重）及可选的 `force_limit_N`（正的控制力绝对值上限）。省略权重时分别使用 `[10,1,100,1]` 和 `1.0`；省略 `force_limit_N` 时不限幅。当前 `configs/examples/lqr.json` 显式使用默认权重并设置 20 N 力上限，例如：

```json
"settings": {"Q_diag": [10, 1, 100, 1], "R": 1.0, "force_limit_N": 20.0}
```

当前 LQR 不包含摆起控制。`configs/lqr_01_*.json` 至 `lqr_17_*.json` 与对应 PID 配置采用相同模型、初态、仿真时长、采样周期和 20 N 力上限，可作为后续控制方法对比的配置起点；性能比较仍需实际运行并选定评价指标。

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

本项目的测试位于 `tests/test_model.py`、`test_simulation.py`、`test_feedback.py`、`test_archive.py`、`test_pid.py` 和 `test_lqr.py`，检查动力学、能量与动量、反馈时序、配置、归档重跑，以及 PID 的方向、两环速度反馈、输出后积分更新和限幅，并核对 17 份 PID 配置除初态外完全一致、覆盖5°至60°的5°步进及其他典型初态。整定测试检查默认工厂实际使用极点而非基准增益、完整五阶极点、模型变化、过大采样周期报错、离散矩阵与真实采样仿真的对应关系，以及代表初态下的非线性恢复。LQR 测试检查反馈增益、离散闭环极点、参数变化及非线性对象上的平衡效果。

PID 测试读取默认配置的参数，并在内存副本中分别选择手填或公式模式，不改工作区的注释选择。手填增益用例检查 10 s 末的响应，公式整定用例检查 20 s 仿真的最后 5 s。初态和目标由测试用例指定，不直接使用 JSON 的 `initial_state` 和 `duration_s`。运行结束显示 `OK` 表示这些测试通过，不等于当前 JSON 初态已经通过仿真验收，也不能由此推断任意配置都能稳定倒立摆。

## 7. 文件导航

| 文件或目录 | 职责 |
| --- | --- |
| `cartpole_model.py` | 非线性动力学、几何、能量、线性状态空间与传递函数 |
| `cartpole_simulation.py` | 积分、反馈时序、单步调试与数据导出 |
| `feedback.py`、`configs/` | 控制器实现入口与运行配置 |
| `run_simulation.py` | 加载配置、组织仿真和结果归档 |
| `analyze_pid_tuning.py` | 极点与限幅的批量对照、诊断和指标汇总 |
| `cartpole_visualization.py` | 绘图和动画 |
| `tests/` | 自动测试（部分测试会运行仿真） |
| `presentation.pptx`、`assets/` | 课堂展示草稿及插图；不是已完成的实验报告 |
| [作业要求.pdf](作业要求.pdf) | 课程要求原件 |
| [AGENTS.md](AGENTS.md) | AI 协作约定与工作交接 |
