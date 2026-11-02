"""Three review layouts for actual PID/LQR noise archives; never simulates.

    python noise_review_plots.py path/to/batch --styles A B C
    python noise_review_plots.py path/to/old-batch --historical-demo --output-dir figures_historical

A: representative initial-angle time envelopes. B: absolute RMS heatmaps
with a common PID/LQR scale plus a signed difference. C: individual paired
seed differences for every initial angle. Missing values remain masked.
Historical data must be explicitly marked and cannot establish performance
of the current-main controllers.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


REVIEW_BASE_COMMIT = "ae4bd98"
METHODS = ("pid", "lqr")
KINDS = ("input", "sensor")
COLORS = {"pid": "#D55E00", "lqr": "#0072B2"}
METRICS = (
    ("tail_angle_rms_deg", "真实摆角 RMS", "°", 1.0),
    ("tail_position_rms_m", "真实位置 RMS", "cm", 100.0),
    ("tail_actuator_rms_N", "执行器力 RMS", "N", 1.0),
)


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    names = {font.name for font in font_manager.fontManager.ttflist}
    candidates = ("Arial Unicode MS", "PingFang SC", "Noto Sans CJK SC", "Microsoft YaHei", "SimHei")
    font = next((name for name in candidates if name in names), "DejaVu Sans")
    plt.rcParams.update({"font.family": [font, "DejaVu Sans"], "font.size": 9,
                         "axes.unicode_minus": False, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42,
                         "savefig.facecolor": "white"})
    return plt


class ReviewBatch:
    def __init__(self, directory, historical_demo=False, expected_base_commit=REVIEW_BASE_COMMIT):
        self.path = Path(directory).resolve()
        self.config = json.loads((self.path / "config.json").read_text(encoding="utf-8"))
        self.manifest = json.loads((self.path / "manifest.json").read_text(encoding="utf-8"))
        self.historical = historical_demo
        self.base_commit = self.config.get("base_commit", "")
        if not historical_demo and (not self.base_commit or not self.base_commit.startswith(expected_base_commit)):
            raise ValueError("Batch source does not match this main review. Use --historical-demo "
                             "for old layout examples, or explicitly set --expected-base-commit.")
        manifest_commit = self.manifest.get("base_commit")
        if manifest_commit and manifest_commit != self.base_commit:
            raise ValueError("config/manifest base commits differ.")
        with (self.path / "summary.csv").open(encoding="utf-8-sig", newline="") as handle:
            self.rows = list(csv.DictReader(handle))
        self.angles = sorted(map(float, self.config["angles_deg"]))
        self.levels = sorted(map(float, self.config["levels"]))
        self.seeds = sorted(map(int, self.config["seeds"]))
        self.duration = float(self.config["duration_s"])
        self.tail = float(self.config["tail_seconds"])
        self.onset = float(self.config["noise_onset_s"])
        self.plot_level = float(self.config.get("plot_level", 1))
        self.index, self.issues, self.cache = {}, [], {}
        for row in self.rows:
            row["angle_deg"] = float(row["angle_deg"])
            row["level"] = float(row["level"])
            row["seed"] = int(row["seed"])
            key = (row["angle_deg"], row["kind"], row["level"], row["seed"], row["method"])
            if key in self.index:
                raise ValueError(f"Duplicate summary record {key}")
            self.index[key] = row
        self.period = self.config.get("sample_period_s")
        if self.period is None:
            for row in self.rows:
                metadata = self.archive(row) / "metadata.json"
                if metadata.is_file():
                    info = json.loads(metadata.read_text(encoding="utf-8"))
                    self.period = info.get("control", {}).get("sample_period_s")
                    if self.period is not None:
                        break
        if self.period is None:
            raise ValueError("Sampling period absent from batch config and run metadata.")
        self.period = float(self.period)
        self.counts = {"planned_noisy_runs": len(self.angles) * len(KINDS) * len(self.levels) * len(self.seeds) * 2,
                       "completed_noisy_runs": 0, "failed_noisy_runs": 0, "missing_noisy_runs": 0}
        for angle in self.angles:
            for kind in KINDS:
                for level in self.levels:
                    for seed in self.seeds:
                        for method in METHODS:
                            row = self.index.get((angle, kind, level, seed, method))
                            field = ("missing_noisy_runs" if row is None else "completed_noisy_runs"
                                     if row.get("run_status") == "completed" else "failed_noisy_runs")
                            self.counts[field] += 1

    def archive(self, row):
        path = (self.path / row["archive"]).resolve()
        if self.path not in path.parents:
            raise ValueError("Archive path escapes the batch directory.")
        return path

    def row(self, angle, kind, level, seed, method):
        return self.index.get((float(angle), kind, float(level), int(seed), method))

    def metric(self, row, field, scale=1):
        if row is None or row.get("run_status") != "completed":
            return np.nan
        try:
            value = float(row[field]) * scale
        except (KeyError, ValueError, TypeError):
            value = np.nan
        if not np.isfinite(value) or value < 0:
            self.issues.append(f"Invalid metric {row.get('archive')} {field}")
            return np.nan
        return value

    def values(self, angle, kind, level, method, field, scale=1):
        values = [self.metric(self.row(angle, kind, level, seed, method), field, scale)
                  for seed in self.seeds]
        return np.asarray(values)

    def paired(self, angle, kind, level, field, scale=1):
        result = []
        for seed in self.seeds:
            pid, lqr = [self.row(angle, kind, level, seed, method) for method in METHODS]
            if not pid or not lqr or any(row.get("run_status") != "completed" for row in (pid, lqr)):
                continue
            digest = pid.get("noise_schedule_sha256", "")
            if len(digest) != 64 or digest != lqr.get("noise_schedule_sha256"):
                self.issues.append(f"Unpaired/mismatched noise hash: {angle}/{kind}/{level}/{seed}")
                continue
            a, b = self.metric(pid, field, scale), self.metric(lqr, field, scale)
            if np.isfinite(a) and np.isfinite(b):
                result.append((seed, a, b))
        return result

    def curve(self, row, filename, field, scale=1):
        if row is None or row.get("run_status") != "completed":
            return None
        key = (row["archive"], filename)
        if key not in self.cache:
            path = self.archive(row) / filename
            if not path.is_file():
                self.issues.append(f"Missing file {row['archive']}/{filename}")
                return None
            self.cache[key] = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))
        data = self.cache[key]
        if not data.dtype.names or field not in data.dtype.names or "time_s" not in data.dtype.names:
            self.issues.append(f"Missing columns {row['archive']}/{filename}/{field}")
            return None
        t, y = data["time_s"], data[field] * scale
        if not np.all(np.isfinite(t)) or not np.all(np.isfinite(y)):
            self.issues.append(f"Nonfinite curve {row['archive']}/{filename}/{field}")
            return None
        return t, y

    def physical_note(self, kind=None):
        distribution = self.config.get("input_distribution", "gaussian")
        sigma = float(self.config["input_std_N"])
        sensor = self.config["sensor_std"]
        text = f"Ts={1000*self.period:g} ms；噪声开始={self.onset:g} s；α为标准差倍数。"
        if kind in (None, "input"):
            text += f" 输入端：{distribution}，基础 σ={sigma:.4g} N"
            if distribution == "uniform":
                text += f"（基础范围 ±{np.sqrt(3)*sigma:.4g} N）"
            text += "。"
        if kind in (None, "sensor"):
            text += (f" 传感器高斯噪声基础 σ=[{sensor[0]:g} m，{sensor[1]:g} m/s，"
                     f"{sensor[2]:g}°，{sensor[3]:g}°/s]。")
        return text

    def footer(self, kind=None):
        c = self.counts
        pilot = "小样本 PILOT；不作显著性结论。" if len(self.seeds) < 10 else "不预设任何控制器优胜。"
        return (f"每工况计划 {len(self.seeds)} 个种子；带噪运行完成 {c['completed_noisy_runs']}/{c['planned_noisy_runs']}，"
                f"失败 {c['failed_noisy_runs']}，缺失 {c['missing_noisy_runs']}。{pilot}\n"
                + self.physical_note(kind))


def _decorate(fig, batch, title, detail, footer, *, top=0.84):
    fig.suptitle(title, fontsize=15, y=0.955 if batch.historical else 0.975)
    if batch.historical:
        fig.text(0.5, 0.985, "HISTORICAL LAYOUT DEMO / OLD PID / not current-main evidence",
                 ha="center", va="top", color="#A12828", fontsize=12, weight="bold",
                 bbox=dict(facecolor="#FFF0EA", edgecolor="#C04D3F", pad=5))
    fig.text(0.5, top + 0.03, detail, ha="center", fontsize=9.2)
    fig.text(0.04, 0.02, footer, fontsize=8.2, va="bottom", linespacing=1.5, color="#46515E")


def _save(fig, directory, stem, plt):
    paths = []
    for extension in ("png", "pdf"):
        path = directory / f"{stem}.{extension}"
        fig.savefig(path, dpi=300)
        paths.append(path)
    plt.close(fig)
    return paths


def layout_a(batch, directory, plt):
    angles = [angle for angle in (5.0, 30.0) if angle in batch.angles]
    if len(angles) != 2:
        angles = sorted(set((batch.angles[0], batch.angles[-1])))
    outputs = []
    for kind in KINDS:
        fig, axes = plt.subplots(3, len(angles), figsize=(12, 9), squeeze=False)
        fig.subplots_adjust(left=0.075, right=0.98, bottom=0.16, top=0.83, hspace=0.34, wspace=0.23)
        for col, angle in enumerate(angles):
            for row_i, (filename, field, scale, label) in enumerate((
                ("trajectory.csv", "angle_deg", 1, "真实摆角 [°]"),
                ("trajectory.csv", "position_m", 100, "真实位置 [cm]"),
                ("noise_control.csv", "actuator_force_N", 1, "执行器力 [N]"),
            )):
                ax = axes[row_i, col]
                zoom = ax.inset_axes([0.53, 0.40, 0.44, 0.37])
                zoom.set_title(f"末 {batch.tail:g} s", fontsize=7, pad=2)
                zoom.tick_params(labelsize=6)
                zoom.grid(alpha=0.12)
                zoom.set_xlim(batch.duration-batch.tail, batch.duration)
                for method in METHODS:
                    curves = []
                    for seed in batch.seeds:
                        record = batch.row(angle, kind, batch.plot_level, seed, method)
                        curve = batch.curve(record, filename, field, scale)
                        if curve is not None:
                            curves.append(curve)
                    if curves:
                        times = curves[0][0]
                        if any(len(t) != len(times) or not np.allclose(t, times, atol=1e-10, rtol=0)
                               for t, _ in curves):
                            raise ValueError("Cannot average different recording time grids.")
                        values = np.stack([y for _, y in curves])
                        mean = values.mean(axis=0)
                        sd = values.std(axis=0, ddof=1) if len(values) > 1 else np.zeros_like(mean)
                        if filename == "noise_control.csv":
                            times = np.r_[times, batch.duration]
                            mean, sd = np.r_[mean, mean[-1]], np.r_[sd, sd[-1]]
                            ax.step(times, mean, where="post", color=COLORS[method], linewidth=1,
                                    label=f"{method.upper()} n={len(curves)}/{len(batch.seeds)}")
                        else:
                            ax.plot(times, mean, color=COLORS[method], linewidth=1.1,
                                    label=f"{method.upper()} n={len(curves)}/{len(batch.seeds)}")
                        ax.fill_between(times, mean-sd, mean+sd, color=COLORS[method], alpha=0.16,
                                        linewidth=0, step="post" if filename == "noise_control.csv" else None)
                        selected = times >= batch.duration-batch.tail-1e-10
                        zoom.plot(times[selected], mean[selected], color=COLORS[method], linewidth=0.8)
                        zoom.fill_between(times[selected], (mean-sd)[selected], (mean+sd)[selected],
                                          color=COLORS[method], alpha=0.16, linewidth=0)
                    else:
                        ax.text(0.03, 0.9 - METHODS.index(method)*0.13,
                                f"{method.upper()} 无可用轨迹", transform=ax.transAxes, color=COLORS[method])
                ax.axvline(batch.onset, color="#555555", linestyle="--", linewidth=0.9)
                ax.axvspan(batch.duration-batch.tail, batch.duration, color="#6A7785", alpha=0.045)
                ax.set_xlim(0, batch.duration)
                ax.set_ylabel(label)
                ax.grid(alpha=0.15)
                if row_i == 0:
                    ax.set_title(f"初始摆角 {angle:g}°；α={batch.plot_level:g}")
                    ax.legend(loc="upper left", bbox_to_anchor=(0.15, 1.0), fontsize=8)
                if row_i == 2:
                    ax.set_xlabel("时间 [s]")
        _decorate(fig, batch, f"A  时序包络｜{'输入端' if kind == 'input' else '传感器'}噪声",
                  "实线=跨种子均值；阴影=±1个样本标准差（非置信区间）；竖虚线=噪声开始",
                  batch.footer(kind) + "\n执行器力不含外部扰动力；末段浅灰区域为RMS统计窗口。", top=0.83)
        outputs += _save(fig, directory, f"noise_A_{kind}", plt)
    return outputs


def _mean(values):
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else np.nan


def layout_b(batch, directory, plt):
    from matplotlib.colors import TwoSlopeNorm
    outputs = []
    for kind in KINDS:
        fig, axes = plt.subplots(3, 3, figsize=(12.5, 9.5))
        fig.subplots_adjust(left=0.08, right=0.96, top=0.84, bottom=0.16, hspace=0.47, wspace=0.33)
        for row_i, (field, label, unit, scale) in enumerate(METRICS):
            grids, counts = {}, {}
            for method in METHODS:
                grid = np.full((len(batch.angles), len(batch.levels)), np.nan)
                count = np.zeros(grid.shape, dtype=int)
                for i, angle in enumerate(batch.angles):
                    for j, level in enumerate(batch.levels):
                        values = batch.values(angle, kind, level, method, field, scale)
                        grid[i, j] = _mean(values)
                        count[i, j] = np.isfinite(values).sum()
                grids[method], counts[method] = grid, count
            difference = np.full_like(grids["pid"], np.nan)
            paired_n = np.zeros(difference.shape, dtype=int)
            for i, angle in enumerate(batch.angles):
                for j, level in enumerate(batch.levels):
                    pairs = batch.paired(angle, kind, level, field, scale)
                    if pairs:
                        difference[i, j] = np.mean([pid-lqr for _, pid, lqr in pairs])
                        paired_n[i, j] = len(pairs)
            absolute = np.r_[grids["pid"].ravel(), grids["lqr"].ravel()]
            finite = absolute[np.isfinite(absolute)]
            maximum = max(float(finite.max()) if finite.size else 1, 1e-12)
            difference_limit = max(float(np.nanmax(np.abs(difference))) if np.any(np.isfinite(difference)) else 1, 1e-12)
            for col, (grid, count, title) in enumerate((
                    (grids["pid"], counts["pid"], "PID"),
                    (grids["lqr"], counts["lqr"], "LQR"),
                    (difference, paired_n, "PID − LQR"))):
                ax = axes[row_i, col]
                cmap = plt.get_cmap("viridis" if col < 2 else "RdBu").copy()
                cmap.set_bad("#D6D9DD")
                kwargs = dict(vmin=0, vmax=maximum) if col < 2 else dict(norm=TwoSlopeNorm(0, -difference_limit, difference_limit))
                picture = ax.imshow(np.ma.masked_invalid(grid), aspect="auto", cmap=cmap, **kwargs)
                ax.set_xticks(range(len(batch.levels)))
                ax.set_xticklabels([f"{value:g}" for value in batch.levels])
                ax.set_yticks(range(len(batch.angles)))
                ax.set_yticklabels([f"{value:g}°" for value in batch.angles])
                ax.set_xlabel("噪声标准差倍数 α")
                ax.set_title(f"{title}｜{label} [{unit}]", fontsize=10)
                if col == 0:
                    ax.set_ylabel("初始摆角")
                for i in range(grid.shape[0]):
                    for j in range(grid.shape[1]):
                        value = grid[i, j]
                        text = f"{value:+.3g}" if col == 2 else f"{value:.3g}"
                        if not np.isfinite(value):
                            text = "缺失"
                        text += f"\nn={count[i,j]}/{len(batch.seeds)}"
                        if np.isfinite(value):
                            rgba = picture.cmap(picture.norm(value))
                            luminance = sum(weight * component for weight, component in
                                            zip((0.2126, 0.7152, 0.0722), rgba[:3]))
                            color = "white" if luminance < 0.49 else "#111111"
                        else:
                            color = "#111111"
                        ax.text(j, i, text, ha="center", va="center", fontsize=7.4, color=color)
                fig.colorbar(picture, ax=ax, fraction=0.045, pad=0.025)
        _decorate(fig, batch, f"B  RMS热图｜{'输入端' if kind == 'input' else '传感器'}噪声",
                  "PID/LQR每行共用色标；差值采用独立对称色标：正/蓝=LQR更低，负/红=PID更低",
                  batch.footer(kind) + f"\n统计末 {batch.tail:g} s；差值仅使用噪声哈希一致的配对种子；灰格=无可用数据，不填0。", top=0.84)
        outputs += _save(fig, directory, f"noise_B_{kind}", plt)
    return outputs


def select_best_c_angle(batch):
    """Post-hoc best case by one declared primary metric; publish every rank."""
    ranking = []
    for angle in batch.angles:
        conditions = []
        for kind in KINDS:
            for level in batch.levels:
                pairs = batch.paired(angle, kind, level, 'tail_angle_rms_deg')
                pid = float(np.mean([p for _, p, _ in pairs])) if pairs else np.nan
                lqr = float(np.mean([l for _, _, l in pairs])) if pairs else np.nan
                relative = float((pid-lqr)/pid) if np.isfinite(pid) and pid > 0 else None
                conditions.append({'kind': kind, 'level': level, 'pairs': len(pairs),
                                   'pid_mean_deg': pid if np.isfinite(pid) else None,
                                   'lqr_mean_deg': lqr if np.isfinite(lqr) else None,
                                   'relative_reduction': relative})
        eligible = all(c['pairs'] == len(batch.seeds) and c['relative_reduction'] is not None for c in conditions)
        score = float(np.mean([c['relative_reduction'] for c in conditions])) if eligible else None
        ranking.append({'angle_deg': angle, 'eligible': eligible,
                        'mean_relative_reduction': score, 'conditions': conditions})
    available = [r for r in ranking if r['eligible']]
    if not available:
        raise ValueError('No complete angle is available for the declared best-C selection rule.')
    best = sorted(available, key=lambda r: (-r['mean_relative_reduction'], r['angle_deg']))[0]
    return {'selected_angle_deg': best['angle_deg'],
            'mean_relative_reduction': best['mean_relative_reduction'],
            'rule': 'Maximize the unweighted mean of (mean_PID-mean_LQR)/mean_PID for true tail angle RMS over every noise type and level; all seed pairs required; smallest angle breaks exact ties.',
            'posthoc_best_case_not_representative': True,
            'positive_average_advantage_observed': best['mean_relative_reduction'] > 0,
            'ranking': ranking}


def layout_c(batch, directory, plt):
    outputs = []
    selection = select_best_c_angle(batch) if batch.config.get('c_best_only', False) else None
    angles = [selection['selected_angle_deg']] if selection else batch.angles
    if selection:
        (directory / 'noise_C_selection.json').write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding='utf-8')
    for angle in angles:
        fig, axes = plt.subplots(2, 3, figsize=(13, 7.4))
        fig.subplots_adjust(left=0.075, right=0.98, top=0.83, bottom=0.20, hspace=0.47, wspace=0.28)
        for row_i, kind in enumerate(KINDS):
            for col, (field, label, unit, scale) in enumerate(METRICS):
                ax = axes[row_i, col]
                all_differences, group_values = [], []
                for j, level in enumerate(batch.levels):
                    pairs = batch.paired(angle, kind, level, field, scale)
                    differences = np.array([pid-lqr for _, pid, lqr in pairs])
                    group_values.append((j, differences))
                    all_differences.extend(differences)
                extent = max(max(map(abs, all_differences), default=1) * 1.45, 1e-12)
                ax.axhspan(0, extent, color=COLORS["lqr"], alpha=0.03)
                ax.axhspan(-extent, 0, color=COLORS["pid"], alpha=0.03)
                ax.axhline(0, color="#666666", linestyle="--", linewidth=0.9)
                for j, values in group_values:
                    offsets = np.linspace(-0.10, 0.10, len(values)) if len(values) > 1 else np.zeros(len(values))
                    ax.scatter(j+offsets, values, s=28, edgecolor="white", linewidth=0.5,
                               c=[COLORS["lqr"] if value >= 0 else COLORS["pid"] for value in values], zorder=4)
                    if len(values):
                        mean = values.mean()
                        ax.plot([j-0.24, j+0.24], [mean, mean], linewidth=3,
                                color=COLORS["lqr"] if mean >= 0 else COLORS["pid"], zorder=3)
                    ax.text(j, -extent*0.85, f"配对 n={len(values)}/{len(batch.seeds)}", ha="center", fontsize=7.5)
                ax.set_xlim(-0.5, len(batch.levels)-0.5)
                ax.set_ylim(-extent, extent)
                ax.set_xticks(range(len(batch.levels)))
                labels = ([f"σ={float(batch.config['input_std_N'])*value:.3g} N" for value in batch.levels]
                          if kind == "input" else [f"α={value:g}" for value in batch.levels])
                ax.set_xticklabels(labels, fontsize=8)
                ax.set_xlabel("输入端噪声" if kind == "input" else "传感器噪声")
                ax.set_ylabel(f"PID − LQR [{unit}]")
                ax.set_title(label)
                ax.grid(axis="y", alpha=0.15)
        selection_note = (f"\n试验后选例：摆角RMS平均改善率最大（{100*selection['mean_relative_reduction']:.2f}%）；不代表全部工况，排名见 noise_C_selection.json。" if selection else '')
        _decorate(fig, batch, f"C  种子配对差｜初始摆角 {angle:g}°" + ('｜最佳案例' if selection else ''),
                  "每点=同种子 PID−LQR；横杠=差值均值；正/蓝=LQR数值低，负/橙=PID数值低；0为持平",
                  batch.footer() + f"\n统计末 {batch.tail:g} s；不同初态不合并样本；只比较噪声序列一致的有效配对。" + selection_note, top=0.83)
        outputs += _save(fig, directory, f"noise_C_{angle:g}deg", plt)
    return outputs


def render_all(batch_dir, styles=("A", "B", "C"), output_dir=None,
               historical_demo=False, expected_base_commit=REVIEW_BASE_COMMIT):
    styles = tuple(str(style).upper() for style in styles)
    if not styles or len(styles) != len(set(styles)) or any(style not in ("A", "B", "C") for style in styles):
        raise ValueError("Choose unique plot styles from A/B/C.")
    batch = ReviewBatch(batch_dir, historical_demo, expected_base_commit)
    directory = Path(output_dir) if output_dir else batch.path / "plots_review"
    directory.mkdir(parents=True, exist_ok=True)
    plt = _pyplot()
    outputs = []
    for style in styles:
        outputs += {"A": layout_a, "B": layout_b, "C": layout_c}[style](batch, directory, plt)
    statistics = []
    for angle in batch.angles:
        for kind in KINDS:
            for level in batch.levels:
                for field, _, unit, scale in METRICS:
                    pairs = batch.paired(angle, kind, level, field, scale)
                    statistics.append({"angle_deg": angle, "kind": kind, "level": level,
                                       "metric": field, "unit": unit,
                                       "pid_mean": _mean(batch.values(angle, kind, level, "pid", field, scale)),
                                       "lqr_mean": _mean(batch.values(angle, kind, level, "lqr", field, scale)),
                                       "paired_mean_pid_minus_lqr": float(np.mean([p-l for _, p, l in pairs])) if pairs else "",
                                       "valid_pairs": len(pairs), "planned_pairs": len(batch.seeds)})
    with (directory / "noise_plot_statistics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(statistics[0]))
        writer.writeheader()
        writer.writerows(statistics)
    audit = {"historical_layout_demo": historical_demo, "current_main_evidence": not historical_demo,
             "base_commit": batch.base_commit, "angles_from_config": batch.angles,
             "summary_sha256": hashlib.sha256((batch.path / "summary.csv").read_bytes()).hexdigest(),
             "no_simulations_run": True, "counts": batch.counts, "issues": sorted(set(batch.issues)),
             "bands": "mean +/-1 sample SD, not confidence intervals", "styles": styles,
             "files": [path.name for path in outputs]}
    (directory / "noise_plot_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    parser.add_argument("--styles", nargs="+", choices=("A", "B", "C"), default=["A", "C"])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--historical-demo", action="store_true")
    parser.add_argument("--expected-base-commit", default=REVIEW_BASE_COMMIT)
    args = parser.parse_args(argv)
    for path in render_all(args.batch, args.styles, args.output_dir, args.historical_demo, args.expected_base_commit):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
