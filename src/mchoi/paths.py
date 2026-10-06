"""Where things live: repository, configuration, models, data and the two Python environments.

Every location can be overridden by an environment variable or by `configs/paths.local.json` (git-ignored; see
`configs/paths.example.json`). Environment variables win over the file, the file over the defaults.

    MCHOI_DATA          data root holding recordings/, sessions/ and the published dataset   (default: <repo>/data)
    MCHOI_MODELS        model folder: wilor/, yolo11x-pose.pt, objects/<ycb>/nontextured.stl (default: <repo>/models)
    MCHOI_MANO          folder with MANO_RIGHT.pkl                                         (default: <models>/mano)
    MCHOI_WILOR_SRC     WiLoR source checkout                                              (default: <repo>/third_party/WiLoR)
    MCHOI_SDK_PYTHON    Python of the SDK environment (ZED SDK, RealSense, OpenCV)         (default: <repo>/.venv-sdk)
    MCHOI_WILOR_PYTHON  Python of the WiLoR environment (PyTorch, WiLoR, Ultralytics)      (default: <repo>/.venv-wilor)
"""
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / 'configs'
ARUCO_DIR = 'aruco_pose_all_cameras_new_calibration'  # per-recording ArUco/timing folder name (kept for compatibility)

_LOCAL = CONFIG_DIR / 'paths.local.json'
_SETTINGS = json.loads(_LOCAL.read_text(encoding='utf-8')) if _LOCAL.exists() else {}


def _setting(env, key, default):
    value = os.environ.get(env) or _SETTINGS.get(key)
    return Path(value) if value else Path(default)


def _venv_python(folder):
    return REPO_ROOT / folder / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')


MODELS = _setting('MCHOI_MODELS', 'models', REPO_ROOT / 'models')
MANO_DIR = _setting('MCHOI_MANO', 'mano', MODELS / 'mano')
WILOR_SRC = _setting('MCHOI_WILOR_SRC', 'wilor_src', REPO_ROOT / 'third_party' / 'WiLoR')


def data_root():
    return _setting('MCHOI_DATA', 'data', REPO_ROOT / 'data')


def sdk_python():
    return _setting('MCHOI_SDK_PYTHON', 'sdk_python', _venv_python('.venv-sdk'))


def wilor_python():
    return _setting('MCHOI_WILOR_PYTHON', 'wilor_python', _venv_python('.venv-wilor'))


def subprocess_env():
    """Environment for child processes: makes `python -m mchoi....` work even without `pip install -e .`, and writes
    their console output as UTF-8 (log files, and non-ASCII text on a Windows console code page)."""
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join(filter(None, [str(REPO_ROOT / 'src'), env.get('PYTHONPATH')]))
    env.setdefault('PYTHONIOENCODING', 'utf-8')
    return env


def dataset_root(config):
    """Output folder of a release: MCHOI_DATASET if set (e.g. a scratch copy for checks), else <data>/<config root>."""
    value = os.environ.get('MCHOI_DATASET')
    return Path(value) if value else data_root() / config['root']


def calibration_path(take):
    """Calibration copied verbatim by the ArUco export."""
    return Path(take) / ARUCO_DIR / 'calibration_used.json'


def timing_path(take):
    """Per-take RealSense timing: the validated wilor_timing file when present, else the ArUco estimate.

    Pass --skip-unvalidated-timing to mchoi.io.stream_four_views with either file so that views whose offset could not
    be validated are left out instead of being matched with an unverified offset.
    """
    take = Path(take)
    validated = take / 'wilor_timing' / 'timing_alignment.json'
    return validated if validated.exists() else take / ARUCO_DIR / 'timing_alignment.json'
