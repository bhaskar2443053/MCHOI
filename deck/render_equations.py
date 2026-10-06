"""Render the deck equations with matplotlib mathtext to measure them (eq_sizes.json; PNGs in deck/eq/, not used in the deck).

These sizes are only the fallback for equations that PowerPoint has not measured yet (eq_measured.json).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams['mathtext.fontset'] = 'cm'
OUT = Path(__file__).parent / 'eq'
OUT.mkdir(exist_ok=True)
DPI, SIZE = 300, 20  # rendered at 20 pt; the deck scales every equation by the same factor

from equations import EQUATIONS

sizes = {}
for name, tex in EQUATIONS.items():
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, tex, fontsize=SIZE, color='#1B1F27')
    path = OUT / f'{name}.png'
    fig.savefig(path, dpi=DPI, transparent=True, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    from PIL import Image
    sizes[name] = Image.open(path).size
(Path(__file__).parent / 'eq_sizes.json').write_text(json.dumps(dict(dpi=DPI, size_pt=SIZE, images=sizes), indent=1))
print({k: v for k, v in sizes.items()})
