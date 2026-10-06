"""Reproduce a release (default MCHOI v1.0) from the rig calibration and the raw recordings.

Stages, in order:
  pipeline   mchoi.pipeline for every recording, with the settings stored in the release config (finished steps are
             skipped, so an interrupted run resumes); one log per recording in <data>/logs/
  summary    per-recording table of the pipeline results (<data>/logs/<run-name>_summary.md)
  videos     standard RGB/depth videos of every camera (mchoi.dataset.videos; recordings already exported are skipped)
  dataset    dataset layout, metadata, calibration, meshes and derivatives (mchoi.dataset.build)
  media      figures and GIFs for the documentation (mchoi.dataset.media)
  deck       pipeline slides in docs/ (deck/build.py; needs Node.js and `npm ci` in deck/)
  checksums  SHA256SUMS.txt over the finished dataset (mchoi.dataset.build --checksums-only)

The dataset is written to <data>/<root in the config>, or to $MCHOI_DATASET when set.

Examples:
  python -m mchoi.reproduce                                   # everything
  python -m mchoi.reproduce --stages pipeline --jobs 3        # process three recordings at a time
  python -m mchoi.reproduce --only mug1_20260923_143947_317314 --dry-run
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

from mchoi.paths import CONFIG_DIR, MODELS, REPO_ROOT, data_root, dataset_root, sdk_python, subprocess_env

STAGES = ('pipeline', 'summary', 'videos', 'dataset', 'media', 'deck', 'checksums')


def pipeline_command(config, rec):
    """mchoi.pipeline arguments for one recording, from its 'pipeline' settings in the release config."""
    p = rec['pipeline']
    cmd = [str(sdk_python()), '-u', '-m', 'mchoi.pipeline', '--take', str(data_root() / 'recordings' / rec['source']),
           '--calibration', str(data_root() / config['calibration']), '--run-name', config['run_name'],
           '--marker-ids', *map(str, p['marker_ids']), '--object-name', p['object_name']]
    if rec['object']:
        mesh = [] if p.get('markers_only') else ['--mesh', str(MODELS / config['objects'][rec['object']]['mesh'])]
        cmd += mesh or ['--markers-only']
        cmd += ['--object-markers', *map(str, rec['markers'])]
    if 'hsv_low' in p:
        cmd += ['--object-hsv-low', *map(str, p['hsv_low']), '--object-hsv-high', *map(str, p['hsv_high'])]
    if p.get('fit_scale'):
        cmd += ['--object-fit-scale', p['fit_scale']]
    if p.get('shoulder_prior'):
        cmd += ['--shoulder-prior', *map(str, p['shoulder_prior'])]
    return cmd


def run(cmd, log=None):
    """Run one command from the repository; output to the console, or to a log file."""
    if log is None:
        return subprocess.run(cmd, cwd=REPO_ROOT, env=subprocess_env()).returncode
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, 'w', encoding='utf-8') as handle:
        return subprocess.run(cmd, cwd=REPO_ROOT, env=subprocess_env(), stdout=handle, stderr=subprocess.STDOUT).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, default=CONFIG_DIR / 'mchoi_v1.0.json')
    parser.add_argument('--stages', nargs='+', choices=STAGES, default=STAGES)
    parser.add_argument('--only', nargs='+', metavar='SOURCE', help='Recordings (source names) for the pipeline and videos stages')
    parser.add_argument('--jobs', type=int, default=1, help='Recordings processed in parallel in the pipeline stage')
    parser.add_argument('--dry-run', action='store_true', help='Print the commands without running them')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    recordings = [r for r in config['recordings'] if not args.only or r['source'] in args.only]
    unknown = set(args.only or ()) - {r['source'] for r in recordings}
    if unknown:
        parser.error(f'not in {args.config.name}: {", ".join(sorted(unknown))}')
    print(f'release:   {config["name"]} {config["version"]}\ndata root: {data_root()}\nmodels:    {MODELS}\n'
          f'dataset:   {dataset_root(config)}', flush=True)

    py, cfg = str(sdk_python()), ['--config', str(args.config.resolve())]
    steps = {
        'summary': [py, '-u', '-m', 'mchoi.dataset.summarize', '--recordings', str(data_root() / 'recordings'),
                    '--run-name', config['run_name'], '--output', str(data_root() / 'logs' / f'{config["run_name"]}_summary.md')],
        'videos': [py, '-u', '-m', 'mchoi.dataset.videos', *cfg] + (['--only', *args.only] if args.only else []),
        'dataset': [py, '-u', '-m', 'mchoi.dataset.build', *cfg],
        'media': [py, '-u', '-m', 'mchoi.dataset.media', *cfg],
        'deck': [py, '-u', str(REPO_ROOT / 'deck' / 'build.py'), *cfg],
        'checksums': [py, '-u', '-m', 'mchoi.dataset.build', *cfg, '--checksums-only'],
    }
    failed = []
    for stage in (s for s in STAGES if s in args.stages):
        print(f'[{stage}]', flush=True)
        if stage == 'pipeline':
            def one(rec):
                cmd = pipeline_command(config, rec)
                if args.dry_run:
                    print('  ' + subprocess.list2cmdline(cmd), flush=True)
                    return 0
                log = data_root() / 'logs' / f'{config["run_name"]}_{rec["source"]}.log'
                code = run(cmd, log)
                print(f'  {rec["source"]}: ' + ('ok' if code == 0 else f'FAILED (exit {code}), see {log}'), flush=True)
                return code

            with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
                codes = list(pool.map(one, recordings))
            failed += [r['source'] for r, code in zip(recordings, codes) if code]
        elif args.dry_run:
            print('  ' + subprocess.list2cmdline(steps[stage]), flush=True)
        elif run(steps[stage]):
            failed.append(stage)
        if failed:
            sys.exit(f'Stopped after {stage}; failed: {", ".join(failed)}')
    print('done')


if __name__ == '__main__':
    main()
