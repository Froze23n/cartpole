"""Summarize complete seed pairs; never pool initial angles as independent seeds."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

from noise_review_plots import ReviewBatch, select_best_c_angle


FIELDS = ['tail_angle_rms_deg', 'tail_position_rms_m', 'tail_actuator_rms_N',
          'tail_angle_abs_p95_deg', 'IAE_angle_rad_s', 'IAE_position_m_s',
          'integral_u2_N2_s', 'actuator_tv_rate_N_s', 'max_abs_position_m',
          'max_abs_angle_deg', 'saturation_fraction']


def summarize(path):
    batch = ReviewBatch(path)
    stats = []
    for angle in batch.angles:
        for kind in ('input', 'sensor'):
            for level in batch.levels:
                for field in FIELDS:
                    pairs = batch.paired(angle, kind, level, field)
                    if len(pairs) != len(batch.seeds):
                        raise ValueError(f'Incomplete pairs: {angle}/{kind}/{level}/{field}')
                    pid = np.array([p for _, p, _ in pairs])
                    lqr = np.array([l for _, _, l in pairs])
                    difference = pid-lqr
                    half = float(t.ppf(0.975, len(pairs)-1)*difference.std(ddof=1)/np.sqrt(len(pairs)))
                    nominal_pid = float(batch.row(angle, 'baseline', 0, batch.seeds[0], 'pid')[field])
                    nominal_lqr = float(batch.row(angle, 'baseline', 0, batch.seeds[0], 'lqr')[field])
                    stats.append({'angle_deg': angle, 'kind': kind, 'level': level, 'metric': field,
                                  'n_pairs': len(pairs), 'pid_mean': float(pid.mean()),
                                  'pid_sample_sd': float(pid.std(ddof=1)),
                                  'lqr_mean': float(lqr.mean()), 'lqr_sample_sd': float(lqr.std(ddof=1)),
                                  'mean_pid_minus_lqr': float(difference.mean()),
                                  'paired_difference_ci95_lower': float(difference.mean()-half),
                                  'paired_difference_ci95_upper': float(difference.mean()+half),
                                  'relative_reduction_from_pid': float((pid.mean()-lqr.mean())/pid.mean()) if pid.mean() else None,
                                  'pid_nominal_baseline': nominal_pid, 'lqr_nominal_baseline': nominal_lqr,
                                  'pid_mean_increment_vs_own_baseline': float(pid.mean()-nominal_pid),
                                  'lqr_mean_increment_vs_own_baseline': float(lqr.mean()-nominal_lqr),
                                  'pid_minus_lqr_increment': float((pid.mean()-nominal_pid)-(lqr.mean()-nominal_lqr)),
                                  'seed_pairs_lqr_lower': int(np.sum(lqr < pid)),
                                  'seed_pairs_pid_lower': int(np.sum(pid < lqr))})
    with (path / 'condition_statistics.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(stats[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(stats)
    benefits = {}
    for kind in ('input', 'sensor'):
        benefits[kind] = {}
        for field in FIELDS:
            subset = [r for r in stats if r['kind'] == kind and r['metric'] == field]
            values = [r['relative_reduction_from_pid'] for r in subset if r['relative_reduction_from_pid'] is not None]
            benefits[kind][field] = {
                'conditions': len(subset),
                'conditions_lqr_mean_lower': sum(r['mean_pid_minus_lqr'] > 0 for r in subset),
                'conditions_pid_mean_lower': sum(r['mean_pid_minus_lqr'] < 0 for r in subset),
                'relative_reduction_min': min(values) if values else None,
                'relative_reduction_max': max(values) if values else None,
                'pointwise_ci_entirely_above_zero': sum(r['paired_difference_ci95_lower'] > 0 for r in subset),
                'pointwise_ci_entirely_below_zero': sum(r['paired_difference_ci95_upper'] < 0 for r in subset)}
    counts = {}
    for kind in ('baseline', 'input', 'sensor'):
        counts[kind] = {}
        for method in ('pid', 'lqr'):
            subset = [r for r in batch.rows if r['kind'] == kind and r['method'] == method]
            counts[kind][method] = {
                'runs': len(subset),
                'completed': sum(r['run_status'] == 'completed' for r in subset),
                'upright_region_held': sum(r['upright_region_held'] == 'True' for r in subset),
                'strict_nominal_tolerance_pass': sum(r['baseline_tolerance_pass'] == 'True' for r in subset),
                'max_abs_position_m': max(float(r['max_abs_position_m']) for r in subset),
                'max_abs_angle_deg': max(float(r['max_abs_angle_deg']) for r in subset),
                'max_peak_actuator_force_N': max(float(r['peak_actuator_force_N']) for r in subset)}
    result = {'base_commit': batch.base_commit, 'total_runs': len(batch.rows), 'counts': counts,
              'condition_comparisons': benefits, 'C_selection': select_best_c_angle(batch),
              'uncertainty_note': '95% intervals are pointwise paired t intervals over 20 seeds within a fixed initial-angle/noise/level condition. They are not simultaneous intervals, do not adjust for post-hoc C selection, and do not pool correlated initial angles or noise levels.',
              'generalization_note': 'Only these fixed controller settings, sampled states, noise distributions, intensities and 30 s horizon were tested. Upright retention is not a proof of global stability or sensor-estimator performance.'}
    (path / 'results_summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch', type=Path)
    summarize(parser.parse_args().batch.resolve())
