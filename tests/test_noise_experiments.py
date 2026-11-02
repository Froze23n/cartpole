"""Static and small-array checks for the noise protocol; no ODE integration.

Run these protocol checks independently of the full experiment:
    python -m unittest discover -s tests -p 'test_noise_experiments.py' -v
"""

from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import run_noise_experiments as noise


ROOT = Path(__file__).resolve().parents[1]
ANGLES = [5, 10, 15, 20, 25, 30]


class NoiseReviewTests(unittest.TestCase):
    def test_best_c_uses_all_conditions_rather_than_a_single_favorable_condition(self):
        from noise_review_plots import select_best_c_angle
        def paired(angle, kind, level, field):
            lqr = (0 if kind == 'input' and level == 0.5 else 0.9) if angle == 5 else 0.6
            return [(seed, 1, lqr) for seed in range(20)]
        batch = SimpleNamespace(angles=[5, 30], levels=[0.5, 1, 2], seeds=list(range(20)), paired=paired)
        result = select_best_c_angle(batch)
        self.assertEqual(result['selected_angle_deg'], 30)
        self.assertAlmostEqual(result['mean_relative_reduction'], 0.4)
        self.assertTrue(result['posthoc_best_case_not_representative'])
        self.assertEqual(len(result['ranking']), 2)

    def test_best_c_excludes_incomplete_angles_and_breaks_exact_ties_by_angle(self):
        from noise_review_plots import select_best_c_angle
        def paired(angle, kind, level, field):
            return [(seed, 1, 0.5) for seed in range(19 if angle == 30 else 20)]
        batch = SimpleNamespace(angles=[30, 10, 5], levels=[0.5, 1, 2], seeds=list(range(20)), paired=paired)
        result = select_best_c_angle(batch)
        self.assertEqual(result['selected_angle_deg'], 5)
        self.assertFalse(result['ranking'][0]['eligible'])

    def test_nominal_baseline_subtraction_keeps_negative_changes_and_missing_values(self):
        rows = [dict(angle_deg=5, method='pid', kind='baseline', run_status='completed', tail_angle_rms_deg=0.2),
                dict(angle_deg=5, method='pid', kind='input', run_status='completed', tail_angle_rms_deg=0.1),
                dict(angle_deg=5, method='lqr', kind='input', run_status='completed', tail_angle_rms_deg=0.3)]
        noise.add_baseline_deltas(rows)
        self.assertEqual(rows[0]['delta_tail_angle_rms_deg'], 0)
        self.assertAlmostEqual(rows[1]['delta_tail_angle_rms_deg'], -0.1)
        self.assertIsNone(rows[2]['delta_tail_angle_rms_deg'])

    def setUp(self):
        self.config = json.loads((ROOT / 'configs/noise_experiments.json').read_text(encoding='utf-8'))
        # Any accidental integration introduced into these checks fails at
        # once, rather than executing an unapproved short experiment.
        self.block_ode = patch.object(noise, 'simulate_feedback',
                                      side_effect=AssertionError('ODE execution is disabled in review tests.'))
        self.block_ode.start()
        self.addCleanup(self.block_ode.stop)

    def _baseline_copy(self, directory):
        root = Path(directory)
        (root / 'configs').mkdir()
        shutil.copyfile(ROOT / 'feedback.py', root / 'feedback.py')
        for method in ('pid', 'lqr'):
            for src in (ROOT / 'configs').glob(f'{method}_*.json'):
                shutil.copyfile(src, root / 'configs' / src.name)
        return root

    def test_default_design_matches_review_scope_and_current_commit(self):
        config = noise.validate_config(self.config)
        self.assertEqual(config['base_commit'], 'ae4bd98d568786a303b7da33c373f8c5e4a83461')
        self.assertEqual(config['angles_deg'], ANGLES)
        self.assertEqual(config['preview_angles_deg'], [5, 30])
        self.assertEqual(config['noise_onset_s'], 0)
        self.assertEqual(config['input_distribution'], 'uniform')
        self.assertEqual(config['sensor_std_units'], ['m', 'm/s', 'deg', 'deg/s'])

    def test_valid_angle_subset_is_preserved_without_mutating_caller(self):
        config = deepcopy(self.config)
        config['angles_deg'] = [10, 20]
        config['preview_angles_deg'] = [20]
        original = deepcopy(config)
        validated = noise.validate_config(config)
        self.assertEqual(validated['angles_deg'], [10, 20])
        self.assertEqual(config, original)

    def test_current_source_pins_match_and_reject_unreviewed_controller_change(self):
        noise.verify_sources(self.config)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copyfile(ROOT / 'feedback.py', root / 'feedback.py')
            config = {'base_source_blobs': {
                'feedback.py': self.config['base_source_blobs']['feedback.py'],
            }}
            noise.verify_sources(config, root)
            with (root / 'feedback.py').open('a', encoding='utf-8') as handle:
                handle.write('\n# Unreviewed change in isolated test fixture.\n')
            with self.assertRaises(ValueError):
                noise.verify_sources(config, root)

    def test_invalid_angles_and_preview_subsets_are_rejected(self):
        cases = [
            {'angles_deg': []}, {'angles_deg': [5, 5]}, {'angles_deg': [5, 35]},
            {'angles_deg': [True, 30]}, {'preview_angles_deg': []},
            {'preview_angles_deg': [5, 5]}, {'preview_angles_deg': [35]},
            {'angles_deg': [5], 'preview_angles_deg': [30]},
            {'preview_seeds': [2**31]}, {'preview_levels': [9]},
            {'plot_level': 9},
            {'preview_levels': [2], 'plot_level': 1},
        ]
        for change in cases:
            with self.subTest(change=change), self.assertRaises(ValueError):
                noise.validate_config({**self.config, **change})

    def test_invalid_units_scales_seeds_and_onset_are_rejected(self):
        cases = [
            ('sensor_std', [1, 2]), ('sensor_std', [1, 2, 3, np.nan]),
            ('sensor_std', [1, 2, 3, -1]),
            ('sensor_std_units', ['m', 'm/s', 'rad', 'rad/s']),
            ('seeds', [0, 0]), ('seeds', [True]), ('seeds', [-1]),
            ('seeds', [2**32]), ('levels', []), ('levels', [-1]),
            ('levels', [np.inf]), ('duration_s', np.nan),
            ('tail_seconds', self.config['duration_s'] + 1),
            ('input_std_N', 0), ('noise_onset_s', -1),
            ('noise_onset_s', self.config['duration_s']),
            ('input_distribution', 'continuous_white_noise'),
        ]
        for key, value in cases:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                noise.validate_config({**self.config, key: value})

    def test_baseline_discovery_uses_real_angles_and_updated_pid(self):
        pairs = noise.baseline_configs()
        self.assertEqual(set(pairs), set(ANGLES))
        for angle in ANGLES:
            for method in ('pid', 'lqr'):
                config = pairs[angle][method]
                self.assertEqual(config['initial_state'], {
                    'position_m': 0, 'velocity_m_s': 0,
                    'angle_deg': angle, 'angular_velocity_deg_s': 0,
                })
                self.assertEqual(config['control']['sample_period_s'], 0.02)
                self.assertEqual(config['control']['settings']['force_limit_N'], 20)
            self.assertEqual(pairs[angle]['pid']['control']['settings']['tuning_poles'],
                             [-1, -1.2, -1.5, -2, -2.5])
        # In current main, case 02 is 10 degrees and case 06 is 30 degrees.
        self.assertEqual(pairs[10]['pid']['initial_state']['angle_deg'], 10)
        self.assertEqual(pairs[30]['pid']['initial_state']['angle_deg'], 30)

    def test_baseline_discovery_rejects_missing_or_duplicate_angle(self):
        for defect in ('missing', 'duplicate'):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as tmp:
                root = self._baseline_copy(tmp)
                target = root / 'configs/lqr_02_angle_10deg.json'
                if defect == 'missing':
                    target.unlink()
                else:
                    # PID is the discovery entry; LQR is selected by its
                    # matching current-main filename after the PID is found.
                    shutil.copyfile(root / 'configs/pid_02_angle_10deg.json',
                                    root / 'configs/pid_99_duplicate_10deg.json')
                with self.assertRaises((ValueError, FileNotFoundError)):
                    noise.baseline_configs(root)

    def test_baseline_discovery_rejects_unfair_pair_or_per_angle_tuning(self):
        changes = [
            ('simulation', 'output_dt_s', 0.02),
            ('parameters', 'm', 1.2),
            ('control', 'sample_period_s', 0.05),
            ('settings', 'force_limit_N', 19),
            ('settings', 'R', 2),
        ]
        for section, field, value in changes:
            with self.subTest(change=(section, field, value)), tempfile.TemporaryDirectory() as tmp:
                root = self._baseline_copy(tmp)
                path = root / 'configs/lqr_06_angle_30deg.json'
                config = json.loads(path.read_text(encoding='utf-8'))
                target = config['control']['settings'] if section == 'settings' else config[section]
                target[field] = value
                path.write_text(json.dumps(config), encoding='utf-8')
                with self.assertRaises(ValueError):
                    noise.baseline_configs(root)

    def test_schedule_onset_and_partial_terminal_hold(self):
        config = {**self.config, 'duration_s': 0.055, 'noise_onset_s': 0.02}
        times, force, sensor, _ = noise.noise_schedule(config, 0.02, 'input', 1, 0)
        np.testing.assert_allclose(times, [0, 0.02, 0.04])
        self.assertEqual(force[0], 0)
        self.assertTrue(np.any(force[1:] != 0))
        np.testing.assert_array_equal(sensor, np.zeros((3, 4)))
        times, force, sensor, _ = noise.noise_schedule(config, 0.02, 'sensor', 1, 0)
        np.testing.assert_array_equal(sensor[0], np.zeros(4))
        self.assertTrue(np.any(sensor[1:] != 0))
        np.testing.assert_array_equal(force, np.zeros(3))
        self.assertLess(times[-1], config['duration_s'])

    def test_pairing_reproducibility_and_scaled_noise_do_not_depend_on_call_order(self):
        config = {**self.config, 'duration_s': 0.2, 'noise_onset_s': 0}
        first = noise.noise_schedule(config, 0.02, 'input', 1, 42)
        noise.noise_schedule(config, 0.02, 'sensor', 1, 7)
        repeated = noise.noise_schedule(config, 0.02, 'input', 1, 42)
        for a, b in zip(first[:3], repeated[:3]):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(first[3], repeated[3])
        for kind, column in [('input', 1), ('sensor', 2)]:
            base = noise.noise_schedule(config, 0.02, kind, 1, 42)
            stronger = noise.noise_schedule(config, 0.02, kind, 2, 42)
            np.testing.assert_array_equal(stronger[column], 2 * base[column])
        clean = noise.noise_schedule(config, 0.02, 'baseline', 0, 42)
        np.testing.assert_array_equal(clean[1], np.zeros(10))
        np.testing.assert_array_equal(clean[2], np.zeros((10, 4)))

    def test_uniform_force_is_calibrated_by_standard_deviation_not_half_width(self):
        config = {**self.config, 'duration_s': 400, 'noise_onset_s': 0,
                  'input_distribution': 'uniform', 'input_std_N': 0.5}
        _, force, _, _ = noise.noise_schedule(config, 0.02, 'input', 2, 123)
        self.assertTrue(np.all(np.abs(force) <= np.sqrt(3)))
        self.assertGreater(np.max(np.abs(force)), 1.5)
        self.assertAlmostEqual(float(force.std()), 1, delta=0.025)
        self.assertAlmostEqual(float(force.mean()), 0, delta=0.025)

    def test_sensor_noise_units_and_independent_random_stream(self):
        config = {**self.config, 'duration_s': 400, 'noise_onset_s': 0}
        _, _, sensor, _ = noise.noise_schedule(config, 0.02, 'sensor', 1, 123)
        expected = np.asarray(config['sensor_std'], dtype=float)
        expected[2:] = np.deg2rad(expected[2:])
        np.testing.assert_allclose(sensor.std(axis=0), expected, rtol=0.025, atol=0)
        altered = {**config, 'input_distribution': 'gaussian', 'input_std_N': 19}
        _, _, repeated, _ = noise.noise_schedule(altered, 0.02, 'sensor', 1, 123)
        np.testing.assert_array_equal(sensor, repeated)

    def test_measurement_copy_and_three_force_channels_are_distinct(self):
        truth = np.array([0.1, 0.2, 0.3, 0.4])
        error = np.array([[0.01, 0.02, 0.03, 0.04]])
        seen = []

        def controller(t, measured):
            seen.append(measured.copy())
            measured[:] = 999  # An impolite controller cannot mutate the log.
            return 20.0

        wrapped = noise.NoisyFeedback(controller, np.array([0]), np.array([0.5]), error)
        total = wrapped(0, truth)
        np.testing.assert_array_equal(seen[0], truth + error[0])
        np.testing.assert_array_equal(truth, [0.1, 0.2, 0.3, 0.4])
        row = wrapped.rows[0]
        np.testing.assert_array_equal(row[1:5], truth)
        np.testing.assert_array_equal(row[5:9], truth + error[0])
        np.testing.assert_array_equal(row[9:13], error[0])
        np.testing.assert_array_equal(row[-3:], [20, 0.5, 20.5])
        self.assertEqual(total, 20.5)  # External force is not clipped to 20 N.

    def test_feedback_rejects_time_mismatch_and_invalid_actuator(self):
        wrapped = noise.NoisyFeedback(lambda t, x: 0, np.array([0]), np.zeros(1), np.zeros((1, 4)))
        with self.assertRaises(RuntimeError):
            wrapped(0.02, np.zeros(4))
        for value in (20.1, -20.1, np.inf, np.nan):
            with self.subTest(value=value), self.assertRaises(ValueError):
                noise.NoisyFeedback(lambda t, x: value, np.array([0]), np.zeros(1),
                                    np.zeros((1, 4)))(0, np.zeros(4))

    def test_metrics_use_truth_and_exact_actuator_hold_durations(self):
        result = SimpleNamespace(time=np.array([0, 0.03, 0.08]),
                                 state=np.zeros((3, 4)), energy_balance_error=np.zeros(3))
        rows = np.zeros((3, 16))
        rows[:, 0] = [0, 0.03, 0.06]
        rows[:, 5:9] = 999  # Bad measurements must not become true-state RMS.
        rows[:, -3] = [1, 2, 3]
        rows[:, -2] = 100
        rows[:, -1] = rows[:, -3] + 100
        metrics = noise.noise_metrics(result, rows, {
            'tail_seconds': 0.05, 'upright_region_limit_deg': 90, 'noise_onset_s': 0,
        })
        self.assertEqual(metrics['tail_angle_rms_deg'], 0)
        self.assertEqual(metrics['tail_position_rms_m'], 0)
        self.assertAlmostEqual(metrics['integral_u2_N2_s'], 0.33)
        self.assertAlmostEqual(metrics['tail_actuator_rms_N'], np.sqrt(6))
        self.assertEqual(metrics['peak_actuator_force_N'], 3)
        self.assertEqual(metrics['saturation_fraction'], 0)
        self.assertAlmostEqual(metrics['actuator_tv_rate_N_s'], 25)
        self.assertTrue(metrics['baseline_tolerance_pass'])
        self.assertEqual(metrics['tail_band_occupancy'], 1)

    def test_declared_upright_region_and_post_onset_integral_use_true_angle(self):
        states = np.zeros((4, 4))
        states[:, 2] = np.deg2rad([30, 60, 90, 180])
        result = SimpleNamespace(time=np.array([0, 1, 2, 3]), state=states,
                                 energy_balance_error=np.zeros(4))
        rows = np.zeros((3, 16))
        rows[:, 0] = [0, 1, 2]
        metrics = noise.noise_metrics(result, rows, {
            'tail_seconds': 1, 'noise_onset_s': 1, 'upright_region_limit_deg': 90,
        })
        self.assertFalse(metrics['upright_region_held'])
        self.assertEqual(metrics['upright_region_first_exit_s'], 2)
        self.assertAlmostEqual(metrics['IAE_angle_rad_s'], np.deg2rad(255))
        self.assertAlmostEqual(metrics['post_onset_IAE_angle_rad_s'], np.deg2rad(210))


if __name__ == '__main__':
    unittest.main()
