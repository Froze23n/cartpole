"""Run archives retain exact configs, replay inputs and failure information."""

from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from run_simulation import main, read_config


class ArchiveTests(unittest.TestCase):
    def test_verbatim_configuration_timestamp_and_archived_feedback_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            source = folder / 'my_feedback.py'
            source.write_text('def create_controller(**kwargs):\n    return lambda t, state: 0.2\n')
            path = folder / 'original.json'
            config = {'simulation': {'duration_s': 0.06},
                      'control': {'sample_period_s': 0.02, 'file': source.name},
                      'output': {'plot': False, 'animation': {'format': 'none'}}}
            original = json.dumps(config, indent=4).encode() + b'\n\n'
            path.write_bytes(original)
            root = folder / 'results'
            with patch('run_simulation.RESULTS_ROOT', root), redirect_stdout(io.StringIO()):
                self.assertEqual(main(['--config', str(path)]), 0)
                first, = root.iterdir()
                self.assertEqual((first / 'config.json').read_bytes(), original)
                meta = json.loads((first / 'metadata.json').read_text())
                self.assertEqual(datetime.fromisoformat(meta['run']['started_at']).strftime('%Y%m%d-%H%M%S-%f'), first.name)
                self.assertEqual(meta['run']['timezone'], 'Asia/Shanghai')
                self.assertEqual(meta['run']['status'], 'completed')
                self.assertNotIn('input_config', meta)
                self.assertNotIn('values_N', meta['force_input'])
                self.assertFalse((first / 'trajectory.npz').exists())
                self.assertFalse((first / '__pycache__').exists())
                source.write_text('raise RuntimeError("Do not load the changed original")\n')
                self.assertEqual(main(['--config', str(first / 'config.json')]), 0)
                second, = [p for p in root.iterdir() if p != first]
                self.assertEqual((second / 'config.json').read_bytes(), original)
                self.assertEqual((second / 'trajectory.csv').read_bytes(), (first / 'trajectory.csv').read_bytes())

    def test_removed_force_modes_rejected_before_archiving(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            path = folder / 'old.json'
            root = folder / 'results'
            for kind in ['constant', 'sine', 'samples', 'csv']:
                for extra in [{}, {'control': {}}]:
                    path.write_text(json.dumps({'force': {'type': kind}, **extra}))
                    with self.subTest(kind=kind, extra=extra):
                        with self.assertRaisesRegex(ValueError, 'force 输入配置已移除'):
                            read_config(path)
                        with patch('run_simulation.RESULTS_ROOT', root), redirect_stderr(io.StringIO()):
                            with self.assertRaises(SystemExit) as error:
                                main(['--config', str(path)])
                        self.assertEqual(error.exception.code, 2)
                        self.assertFalse(root.exists())

    def test_removed_output_fields_rejected_before_archiving(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            path = folder / 'old.json'
            root = folder / 'results'
            for key, value in [('directory', 'elsewhere'), ('timestamp_subdirectory', False)]:
                path.write_text(json.dumps({'control': {}, 'output': {key: value}}))
                with self.subTest(key=key):
                    with self.assertRaisesRegex(ValueError, f'Unknown keys in output:.*{key}'):
                        read_config(path)
                    with patch('run_simulation.RESULTS_ROOT', root), redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit) as error:
                            main(['--config', str(path)])
                    self.assertEqual(error.exception.code, 2)
                    self.assertFalse(root.exists())

    def test_failed_simulation_keeps_config_and_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            path = folder / 'invalid.json'
            path.write_text(json.dumps({'simulation': {'duration_s': -1},
                                        'control': {'sample_period_s': 0.02,
                                                    'file': str(Path(__file__).resolve().parents[1] / 'feedback.py'),
                                                    'factory': 'create_lqr_controller'},
                                        'output': {'plot': False}}))
            with patch('run_simulation.RESULTS_ROOT', folder / 'results'), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main(['--config', str(path)])
            self.assertEqual(error.exception.code, 2)
            output, = (folder / 'results').iterdir()
            self.assertEqual((output / 'config.json').read_bytes(), path.read_bytes())
            self.assertEqual(json.loads((output / 'metadata.json').read_text())['run']['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
