"""Plot and animate saved nonlinear simulation results; no re-simulation."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
import numpy as np


def plot_result(result, path):
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True, constrained_layout=True)
    fig.suptitle('Nonlinear cart-pole | sampled feedback', fontsize=15)
    panels = [(result.position, 'Cart position p [m]', '#176b8c'),
              (result.velocity, 'Cart velocity v [m/s]', '#176b8c'),
              (np.rad2deg(result.angle), 'Pole angle theta [deg, unwrapped]', '#b45729'),
              (np.rad2deg(result.angular_velocity), 'Angular velocity omega [deg/s]', '#b45729')]
    for ax, (values, label, color) in zip(axes.flat, panels):
        ax.plot(result.time, values, color=color, linewidth=1.4)
        ax.set_ylabel(label)
    ax = axes[2, 0]
    times = np.r_[result.control_log[:, 0], result.time[-1]]
    forces = np.r_[result.control_log[:, -1], result.control_log[-1, -1]]
    ax.step(times, forces, where='post', color='#6951a1')
    ax.set_ylabel('Applied force u [N]')
    axes[2, 1].plot(result.time, result.energy - result.energy[0], label='E(t) - E(0)', color='#22836d')
    axes[2, 1].plot(result.time, result.work, '--', label='Input work', color='#c38125')
    axes[2, 1].set_ylabel('Energy change / work [J]')
    axes[2, 1].legend(loc='best')
    axes[2, 1].set_title(f'Max energy balance error: {np.max(np.abs(result.energy_balance_error)):.2e} J', fontsize=10)
    for ax in axes.flat:
        ax.grid(alpha=0.2)
        ax.set_xlim(0, result.time[-1])
    for ax in axes[-1]:
        ax.set_xlabel('Time [s]')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return Path(path)


def animate_result(result, path, fps=20, playback_speed=1.0):
    """Stream an MP4, or write a GIF, at a constant playback rate.

    A following camera keeps the pole legible even with large cart travel.
    Rendering interpolates saved states; this does not change simulation data.
    """
    if not isinstance(fps, int) or isinstance(fps, bool) or fps <= 0:
        raise ValueError('fps must be a positive integer.')
    if not np.isfinite(playback_speed) or playback_speed <= 0:
        raise ValueError('playback_speed must be finite and positive.')
    path = Path(path)
    if path.suffix.lower() not in ('.mp4', '.gif'):
        raise ValueError('Animation output must end in .mp4 or .gif.')
    L = result.params.length
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.set_aspect('equal', adjustable='box')
    ax.set_ylim(-1.35 * L, 1.55 * L)
    ax.set_xlabel('World horizontal position [m]')
    ax.set_ylabel('Height above pivot [m]')
    ax.set_title('Nonlinear cart-pole | camera follows cart')
    ax.grid(alpha=0.15)
    ax.axhline(-0.23 * L, color='#a0aab4', lw=1)
    cart = Rectangle((0, 0), 0.5 * L, 0.18 * L, facecolor='#176b8c', zorder=3)
    wheels = [Circle((0, 0), 0.04 * L, color='#344354', zorder=4) for _ in range(2)]
    rod, = ax.plot([], [], color='#344354', lw=3, zorder=4)
    bob = Circle((0, L), 0.065 * L, facecolor='#e38c35', zorder=5)
    arrow = FancyArrowPatch((0, 0), (0, 0), arrowstyle='-|>', mutation_scale=15, color='#b45729', lw=2, zorder=6)
    for patch in [cart, *wheels, bob, arrow]:
        ax.add_patch(patch)
    status = ax.text(0.02, 0.98, '', transform=ax.transAxes, va='top', fontsize=10,
                     bbox={'facecolor': 'white', 'alpha': 0.85, 'edgecolor': 'none'})
    frame_count = max(2, int(np.ceil(result.time[-1] / playback_speed * fps)) + 1)
    frame_times = np.minimum(np.arange(frame_count) * playback_speed / fps, result.time[-1])
    frame_states = np.column_stack([np.interp(frame_times, result.time, result.state[:, k]) for k in range(4)])
    tolerance = 8 * np.finfo(float).eps * result.time[-1]
    indices = np.searchsorted(result.control_log[:, 0], frame_times + tolerance, side='right') - 1
    frame_forces = result.control_log[indices, -1]
    force_scale = max(float(np.max(np.abs(frame_forces))), 1e-12)

    def update(index):
        p, v, theta, omega = frame_states[index]
        u = frame_forces[index]
        tip = (p + L * np.sin(theta), L * np.cos(theta))
        cart.set_xy((p - 0.25 * L, -0.18 * L))
        for wheel, offset in zip(wheels, (-0.16 * L, 0.16 * L)):
            wheel.center = (p + offset, -0.19 * L)
        rod.set_data([p, tip[0]], [0, tip[1]])
        bob.center = tip
        arrow.set_positions((p, -0.09 * L), (p + 0.75 * L * u / force_scale, -0.09 * L))
        arrow.set_visible(abs(u) > 1e-12)
        ax.set_xlim(p - 2.2 * L, p + 2.2 * L)
        status.set_text(f't = {frame_times[index]:.2f} s    u = {u:+.3f} N\n'
                        f'p = {p:+.3f} m    v = {v:+.3f} m/s\n'
                        f'theta = {np.rad2deg(theta):+.1f} deg    omega = {np.rad2deg(omega):+.1f} deg/s')
        return [cart, *wheels, rod, bob, arrow, status]

    animation = FuncAnimation(fig, update, frames=frame_count, interval=1000 / fps, blit=False)
    if path.suffix.lower() == '.mp4':
        import matplotlib
        ffmpeg = Path(sys.prefix) / 'bin' / 'ffmpeg'
        if ffmpeg.exists():
            matplotlib.rcParams['animation.ffmpeg_path'] = str(ffmpeg)
        if not FFMpegWriter.isAvailable():
            plt.close(fig)
            raise RuntimeError('FFmpeg is unavailable; activate the cartpole environment or export GIF.')
        writer = FFMpegWriter(fps=fps, codec='libx264', extra_args=['-pix_fmt', 'yuv420p'])
    else:
        writer = PillowWriter(fps=fps)
    try:
        animation.save(path, writer=writer, dpi=100)
    finally:
        plt.close(fig)
    return path
