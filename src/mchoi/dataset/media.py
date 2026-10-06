"""Make the figures and GIFs used by the dataset documentation and pipeline slides (docs/media in the dataset).

Snapshots come from the real pipeline outputs of one example recording (default: mustard, ZED frame 1530) plus the
calibration session. Depth panels need the exported depth videos (mchoi.dataset.videos).
"""
import argparse
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import av
import cv2
import numpy as np
from PIL import Image

from mchoi.paths import ARUCO_DIR, CONFIG_DIR, calibration_path, data_root, dataset_root, timing_path
from mchoi.io.zed import read_exact

CAMS = ('zed_left', 'zed_right', 'd435', 'd455')
LABELS = ('ZED Mini left', 'ZED Mini right', 'RealSense D435 #1', 'RealSense D435 #2')
# Headless Chromium-based browser for the 3D-viewer screenshots ('viewer' and 'gifs' media).
EDGE = Path(os.environ.get('MCHOI_BROWSER', 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'))


def video_frame(path, index):
    with av.open(str(path)) as c:
        for i, f in enumerate(c.decode(video=0)):
            if i == index:
                return f.to_ndarray(format='bgr24')
    raise IndexError(index)


def video_frames(path, start, stop, step):
    out = []
    with av.open(str(path)) as c:
        for i, f in enumerate(c.decode(video=0)):
            if i >= stop:
                break
            if i >= start and (i - start) % step == 0:
                out.append(f.to_ndarray(format='bgr24'))
    return out


def label(img, text, scale=1.):
    cv2.rectangle(img, (0, 0), (int(22 * len(text) * scale) + 20, int(48 * scale)), (0, 0, 0), -1)
    cv2.putText(img, text, (10, int(34 * scale)), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def grid(images, cols, size):
    tiles = [cv2.resize(i, size, interpolation=cv2.INTER_AREA) for i in images]
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    return np.vstack([np.hstack(tiles[k:k + cols]) for k in range(0, len(tiles), cols)])


def save_gif(frames, path, fps, width):
    pil = []
    for f in frames:
        h = int(f.shape[0] * width / f.shape[1])
        pil.append(Image.fromarray(cv2.cvtColor(cv2.resize(f, (width, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)))
    pil = [p.quantize(colors=192, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for p in pil]
    pil[0].save(path, save_all=True, append_images=pil[1:], duration=int(1000 / fps), loop=0, optimize=True)
    print(f'{path.name}: {len(pil)} frames, {path.stat().st_size / 1e6:.1f} MB', flush=True)


def raw_views(take, frame):
    p = subprocess.Popen([sys.executable, '-u', '-m', 'mchoi.io.stream_four_views', '--take', str(take), '--calibration', str(calibration_path(take)),
                          '--timing', str(timing_path(take)), '--start-frame', str(frame), '--max-frames', '1', '--skip-unvalidated-timing'],
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    pipe = p.stdout
    read_exact(pipe, 4)

    def msg():
        n, = struct.unpack('<I', read_exact(pipe, 4))
        return json.loads(read_exact(pipe, n))
    msg()
    g = msg()
    images = []
    for cam in CAMS:
        images.append(np.frombuffer(read_exact(pipe, 1920 * 1080 * 3), np.uint8).reshape(1080, 1920, 3).copy() if cam in g['matches']
                      else np.zeros((1080, 1920, 3), np.uint8))
    p.wait()
    return images


def colour_depth(depth, near, far):
    d = depth.astype(np.float32)
    valid = d > 0
    norm = np.clip((d - near) / (far - near), 0, 1)
    img = cv2.applyColorMap((255 * (1 - norm)).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[~valid] = 0
    return img


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG_DIR / 'mchoi_v1.0.json')
    parser.add_argument('--source', default='mustard_20260923_154513_076645')
    parser.add_argument('--frame', type=int, default=1530)
    parser.add_argument('--only', nargs='+', help='Subset of: calibration raw depth aruco hand arm meshfit overlay viewer all gifs')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    root = dataset_root(config)
    media = root / 'docs/media'
    media.mkdir(parents=True, exist_ok=True)
    take = data_root() / 'recordings' / args.source
    run = take / config['run_name']
    rec = next(r for r in config['recordings'] if r['source'] == args.source)
    prefix = f"sub-{config['subject']}_ses-{config['session']}_task-{rec['task']}_run-{rec['run']}"
    want = lambda k: not args.only or k in args.only
    f = args.frame

    if want('calibration'):
        session = data_root() / config['calibration_session']
        poses = sorted(p for p in session.glob('pose_*') if p.is_dir())
        example = poses[len(poses) // 2]
        info = json.loads((example / 'pose.json').read_text())['views']
        tiles = []
        for cam, name in zip(CAMS, LABELS):
            img = cv2.imread(str(example / f'{cam}.png'))
            for x, y in info.get(cam, {}).get('corners', []):
                cv2.circle(img, (int(x), int(y)), 9, (0, 255, 0), -1, cv2.LINE_AA)
            tiles.append(label(img, name, 3))
        cv2.imwrite(str(media / 'step01_calibration_charuco.jpg'), grid(tiles, 2, (960, 540)))
        cover = []
        for cam, name in zip(CAMS, LABELS):
            base = cv2.imread(str(poses[0] / f'{cam}.png'))
            overlay = base.copy()
            for p in poses:
                pts = np.array(json.loads((p / 'pose.json').read_text())['views'].get(cam, {}).get('corners', []), np.int32)
                if len(pts) >= 3:
                    cv2.fillConvexPoly(overlay, cv2.convexHull(pts), (0, 200, 255))
            img = cv2.addWeighted(overlay, .35, base, .65, 0)
            cover.append(label(img, f'{name}: {len(poses)} board poses', 3))
        cv2.imwrite(str(media / 'step01_calibration_coverage.jpg'), grid(cover, 2, (960, 540)))

    if want('raw'):
        images = raw_views(take, f)
        cv2.imwrite(str(media / 'step02_raw_four_views.jpg'), grid([label(i.copy(), n, 3) for i, n in zip(images, LABELS)], 2, (960, 540)))

    if want('depth'):
        video = root / f"sub-{config['subject']}/ses-{config['session']}/video"
        zed = video_frame(video / f'{prefix}_cam-zedminileft_depth.mkv', f) if (video / f'{prefix}_cam-zedminileft_depth.mkv').exists() else None
        if zed is not None:
            with av.open(str(video / f'{prefix}_cam-zedminileft_depth.mkv')) as c:
                zd = next(fr for i, fr in enumerate(c.decode(video=0)) if i == f).to_ndarray()
            rgb = video_frame(video / f'{prefix}_cam-zedminileft_rgb.mp4', f)
            panels = [label(rgb, 'ZED left RGB', 1.6), label(colour_depth(zd, 300, 1200), 'ZED depth (SDK, mm)', 1.6)]
            for cam in ('d435n1', 'd435n2'):
                tsv = video / f'{prefix}_cam-{cam}_depth_frames.tsv'
                rows = np.genfromtxt(tsv, delimiter='\t', names=True, dtype=None, encoding='utf-8')
                zstamp = np.genfromtxt(video / f'{prefix}_cam-zedminileft_rgb_frames.tsv', delimiter='\t', names=True, dtype=None, encoding='utf-8')['timestamp_ns'][f]
                key = 'zed_aligned_timestamp_ns' if 'zed_aligned_timestamp_ns' in rows.dtype.names else 'timestamp_ns'
                k = int(np.argmin(np.abs(rows[key].astype(np.int64) - zstamp)))
                with av.open(str(video / f'{prefix}_cam-{cam}_depth.mkv')) as c:
                    dd = next(fr for i, fr in enumerate(c.decode(video=0)) if i == k).to_ndarray()
                panels.append(label(colour_depth(dd, 300, 1500), f'{cam} depth (Z16)', 1.0))
            cv2.imwrite(str(media / 'step02_depth_streams.jpg'), grid(panels, 2, (960, 540)))

    if want('aruco'):
        # Synchronized four-view marker overview (same ZED frame in every panel, pose table underneath).
        cv2.imwrite(str(media / 'step03_aruco_markers.jpg'), video_frame(take / ARUCO_DIR / 'all_cameras_annotated.mp4', f))

    if want('hand'):
        pick = lambda pattern: sorted((run / pattern.split('/')[0]).glob(pattern.split('/')[1]))[3]
        cv2.imwrite(str(media / 'step04_hand_detection.jpg'), cv2.imread(str(pick('hand_cache/views_*.jpg'))))
        cv2.imwrite(str(media / 'step05_mano_fit.jpg'), cv2.imread(str(pick('hand_fit_one_euro/left_*.jpg'))))
        cv2.imwrite(str(media / 'step05_filter_closeup.jpg'), cv2.imread(str(pick('hand_fit_one_euro/closeup_*.jpg'))))

    if want('arm'):
        cv2.imwrite(str(media / 'step06_right_arm.jpg'), video_frame(run / 'right_arm/right_arm_one_euro.mp4', f))

    if want('meshfit'):
        sheet = cv2.imread(str(next((run / 'mocap').glob('*_mesh_to_marker*.jpg'))))
        cv2.imwrite(str(media / 'step07_mesh_fit.jpg'), sheet[:540])

    if want('overlay'):
        cv2.imwrite(str(media / 'step09_overlay.jpg'), video_frame(next((run / 'mocap').glob('*_overlay.mp4')), f))

    if want('viewer') and not EDGE.exists():
        print(f'[skip] 3D-viewer screenshots: no browser at {EDGE} (set MCHOI_BROWSER)', flush=True)
    if want('viewer') and EDGE.exists():
        html = next((run / 'mocap').glob('*_mocap_viewer.html'))
        rows = [int(l.split('|', 3)[1]) for l in open(next((run / 'mocap').glob('*_Mocap.txt'))) if l[:1].isdigit()]
        row = rows.index(f) if f in rows else len(rows) // 2
        shot = media / 'step09_viewer.png'
        subprocess.run([str(EDGE), '--headless=new', '--disable-gpu', '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
                        '--window-size=1400,850', '--virtual-time-budget=10000', f'--screenshot={shot}',
                        html.resolve().as_uri() + f'#i={row}'], capture_output=True, timeout=120)
        if want('gifs'):
            frames = []
            for r in range(max(0, row - 60), min(len(rows), row + 60), 4):
                tmp = media / '_viewer_tmp.png'
                subprocess.run([str(EDGE), '--headless=new', '--disable-gpu', '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
                                '--window-size=1000,620', '--virtual-time-budget=8000', f'--screenshot={tmp}',
                                html.resolve().as_uri() + f'#i={r}'], capture_output=True, timeout=120)
                img = cv2.imread(str(tmp))
                frames.append(img[:560])
            tmp.unlink(missing_ok=True)
            save_gif(frames, media / 'anim_3d_viewer.gif', 8, 760)

    if want('all') or want('gifs'):
        panels, clips = [], []
        for r in config['recordings']:
            mrun = data_root() / 'recordings' / r['source'] / config['run_name'] / 'mocap'
            txt = next(mrun.glob('*_Mocap.txt'))
            rws = [int(l.split('|', 3)[1]) for l in open(txt) if l[:1].isdigit()]
            mid = rws[int(len(rws) * .45)]
            mp4 = next(mrun.glob('*_overlay.mp4'))
            if want('gifs'):
                clip = [fr[:540, :960] for fr in video_frames(mp4, mid, mid + 60, 3)]
                clips.append([label(c.copy(), f"{r['task']} run-{r['run']}", .9) for c in clip])
            if want('all'):
                panels.append(label(video_frame(mp4, mid)[:540, :960].copy(), f"{r['task']} run-{r['run']}", .9))
        if panels:
            cv2.imwrite(str(media / 'dataset_all_recordings.jpg'), grid(panels, 3, (640, 360)))
        if clips:
            n = min(len(c) for c in clips)
            save_gif([grid([c[k] for c in clips], 3, (320, 180)) for k in range(n)], media / 'anim_all_recordings.gif', 10, 960)

    if want('gifs'):
        mp4 = next((run / 'mocap').glob('*_overlay.mp4'))
        save_gif(video_frames(mp4, f - 60, f + 60, 2), media / 'anim_overlay_four_views.gif', 15, 800)


if __name__ == '__main__':
    main()
