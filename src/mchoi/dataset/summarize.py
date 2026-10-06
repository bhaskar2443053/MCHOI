"""Summarize pipeline runs across recordings: frames, arm/object sources, mesh fit, jitter; optional overlay contact sheet.

Writes a Markdown table (and JSON) of every <take>/<run-name> found under the recordings folder.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def summarize(run):
    out = dict(take=run.parent.name)
    mocap = sorted(run.glob('mocap/*_Mocap.json'))
    if not mocap:
        return None
    m = json.loads(mocap[0].read_text())
    out.update(mocap=str(mocap[0].relative_to(run.parent.parent)), frames=m['frames_total'], written=m['frames_written'],
               no_hand=m['skipped']['no_hand'], no_arm=m['skipped']['no_arm'], with_object=m['frames_with_object'])
    obj = m.get('object', {})
    out['object'] = obj.get('mesh') if obj.get('mesh') not in (None, 'none') else None
    jitter = obj.get('primary_pose_jitter', {})
    if jitter:
        out['object_jitter_p95_mm'] = (round(jitter['raw']['position_mm_p95'], 1), round(jitter['filtered']['position_mm_p95'], 1))
    fits = sorted(run.glob('mocap/*_mesh_to_marker*.json'))
    if fits:
        f = json.loads(fits[0].read_text())
        out.update(mesh_iou=round(f['mean_iou'], 2), rest_tilt_deg=round(f['rest_tilt_deg'], 1) if f.get('rest_tilt_deg') is not None else None,
                   scale=[round(v, 2) for v in f.get('scale', [1, 1, 1])], marker_in_mesh_cm=[round(v * 100, 1) for v in f['marker_origin_in_mesh_m']])
    raw = run / 'right_arm/arm3d_raw.npz'
    if raw.exists():
        method = np.load(raw)['method']
        hand = method[:, 2] > 0
        out['shoulder_observed'] = round(float(np.isin(method[hand, 0], (1, 2)).mean()), 2) if hand.any() else 0.
        out['elbow_observed'] = round(float(np.isin(method[hand, 1], (1, 2)).mean()), 2) if hand.any() else 0.
    arm = run / 'right_arm/arm3d_one_euro.json'
    if arm.exists():
        j = json.loads(arm.read_text())['jitter_mm']
        out['elbow_jitter_mm'] = tuple(None if v is None else round(v, 1) for v in (j['elbow']['raw'], j['elbow']['filtered']))
    timing = run.parent / 'wilor_timing/timing_alignment.json'
    if timing.exists():
        out['views'] = json.loads(timing.read_text()).get('active_cameras')
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recordings', type=Path, default=Path('recordings'))
    parser.add_argument('--run-name', default='pipeline')
    parser.add_argument('--output', type=Path, help='.md path (a .json is written next to it)')
    args = parser.parse_args()
    rows = [r for r in (summarize(run) for run in sorted(args.recordings.glob(f'*/{args.run_name}'))) if r]
    head = '| Take | Views | Frames written / total | Shoulder seen | Object | Mesh fit (IoU, rest tilt, scale) | Object p95 jump raw→filtered |'
    lines = [head, '| --- | --- | --- | --- | --- | --- | --- |']
    for r in rows:
        views = ', '.join(v.replace('zed_', 'ZED ') for v in r.get('views') or []) or 'all 4'
        fit = f"{r['mesh_iou']}, {r['rest_tilt_deg']}°, ×{'/'.join(map(str, r['scale'])) if len(set(r['scale'])) > 1 else r['scale'][0]}" if 'mesh_iou' in r else '—'
        jit = '{} → {} mm'.format(*r['object_jitter_p95_mm']) if 'object_jitter_p95_mm' in r else '—'
        lines.append(f"| {r['take']} | {views} | {r['written']} / {r['frames']} | {int(100 * r.get('shoulder_observed', 0))}% | "
                     f"{r['object'] or '—'} | {fit} | {jit} |")
    table = '\n'.join(lines)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(table + '\n', encoding='utf-8')
        args.output.with_suffix('.json').write_text(json.dumps(rows, indent=2))
    print(table)


if __name__ == '__main__':
    main()
