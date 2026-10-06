"""Build the pipeline slides (docs/MCHOI_v1.0_pipeline.pptx in the dataset) with native, editable PowerPoint equations.

  1. --equations: equations.py (LaTeX) -> eq_omml.json (OMML + plain-text fallback; needs Microsoft Office's
     MML2OMML.XSL) and eq_sizes.json (matplotlib metrics, fallback sizes)
  2. build_deck.js writes the slides with one placeholder text box per equation (Node.js; run `npm ci` in deck/ once)
  3. inject_equations.py replaces the placeholders with OMML equations
  4. --measure: PowerPoint measures every equation (eq_measured.json), then steps 2-3 are repeated with those sizes

The generated eq_*.json files are committed, so a plain build needs neither Office nor PowerPoint.

Usage: python deck/build.py [--output deck.pptx] [--equations] [--measure]
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

from lxml import etree

from mchoi.paths import CONFIG_DIR, dataset_root

HERE = Path(__file__).resolve().parent


def build(dataset, out):
    node = shutil.which('node')
    if not node:
        sys.exit('Node.js not found (https://nodejs.org)')
    if not (HERE / 'node_modules' / 'pptxgenjs').exists():
        sys.exit(f'pptxgenjs is not installed: run `npm ci` in {HERE}')
    env = dict(os.environ, MCHOI_DATASET_DIR=str(dataset), DECK_OUT=str(out))
    subprocess.run([node, str(HERE / 'build_deck.js')], cwd=HERE, env=env, check=True)
    subprocess.run([sys.executable, str(HERE / 'inject_equations.py'), str(out)], check=True)


def check(out):
    """Every XML part parses and no equation placeholder is left."""
    with zipfile.ZipFile(out) as z:
        slides = 0
        for name in z.namelist():
            if name.endswith(('.xml', '.rels')):
                data = z.read(name)
                etree.fromstring(data)
                if name.startswith('ppt/slides/slide'):
                    slides += 1
                    if '⟦EQ:'.encode() in data:
                        sys.exit(f'{name}: equation placeholder left')
    print(f'{out}: {slides} slides, all parts well-formed')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, default=CONFIG_DIR / 'mchoi_v1.0.json')
    parser.add_argument('--dataset', type=Path, help='Dataset folder with docs/media (default: from the release config)')
    parser.add_argument('--output', type=Path, help='Default: <dataset>/docs/<name>_pipeline.pptx')
    parser.add_argument('--equations', action='store_true', help='Regenerate eq_omml.json and eq_sizes.json from equations.py')
    parser.add_argument('--measure', action='store_true', help='Measure the equations with PowerPoint (Windows) and rebuild')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    dataset = (args.dataset or dataset_root(config)).resolve()
    out = (args.output or dataset / 'docs' / f"{config['name']}_v{config['version']}_pipeline.pptx").resolve()
    if not (dataset / 'docs' / 'media').is_dir():
        sys.exit(f'{dataset / "docs/media"} not found: run python -m mchoi.dataset.media first')
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.equations:
        subprocess.run([sys.executable, str(HERE / 'render_equations.py')], cwd=HERE, check=True)
        subprocess.run([sys.executable, str(HERE / 'make_omml.py')], cwd=HERE, check=True)
    build(dataset, out)
    if args.measure:
        subprocess.run(['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(HERE / 'measure_equations.ps1'),
                        '-Deck', str(out), '-Out', str(HERE / 'eq_measured.json')], check=True)
        build(dataset, out)
    check(out)


if __name__ == '__main__':
    main()
