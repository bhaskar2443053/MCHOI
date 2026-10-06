"""Export camera RGB/depth videos (mchoi.dataset.camera_videos) for every recording in the release config."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from mchoi.paths import CONFIG_DIR, REPO_ROOT, data_root, dataset_root, subprocess_env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG_DIR / 'mchoi_v1.0.json')
    parser.add_argument('--only', nargs='+', help='Source recording names to process (default: all)')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    root = dataset_root(config)
    labels = ':'.join([config['camera_labels']['zed']] + [f'{s}={l}' for s, l in config['camera_labels'].items() if s != 'zed'])
    for rec in config['recordings']:
        if args.only and rec['source'] not in args.only:
            continue
        sub, ses = f"sub-{config['subject']}", f"ses-{config['session']}"
        prefix = f"{sub}_{ses}_task-{rec['task']}_run-{rec['run']}"
        out = root / sub / ses / 'video'
        if list(out.glob(f'{prefix}_cam-*')):
            print(f'[skip] {prefix}', flush=True)
            continue
        print(f'[run ] {prefix} <- {rec["source"]}', flush=True)
        subprocess.run([sys.executable, '-u', '-m', 'mchoi.dataset.camera_videos', str(data_root() / 'recordings' / rec['source']),
                        '--output-dir', str(out), '--prefix', prefix, '--labels', labels], cwd=REPO_ROOT, env=subprocess_env(), check=True)


if __name__ == '__main__':
    main()
