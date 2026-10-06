"""Build a self-contained 3D HTML player for a version-4 Mocap text file (arm/hand bones, bone axes, object mesh).

The viewer reads only what is in the Mocap file, so it also checks the file: the object mesh is placed with the
documented rule vertex v (m) -> LOC + ROT * 100 * (SX v.x, -SY v.y, SZ v.z) in world cm. World axes are Unreal (X forward,
Y right, Z up, left-handed); the page displays them by swapping Y and Z into three.js' right-handed Y-up frame.
The STL is reduced by vertex clustering (--cell-mm) to keep the page small.

With --fit, the fitted MANO hand mesh is added: its vertices go through the same world transform as the exported bones
(from the export sidecar .json), stored relative to the wrist at 0.1 mm as int16 frame-to-frame deltas, gzip-compressed,
and decoded in the browser. The arm mesh is procedural: tapered limbs between the mocap joints, with the forearm
narrowing to an elliptical wrist aligned with hand_r, so it shows forearm twist; it is not a scanned body.
"""
import argparse
import base64
import gzip
import json
from pathlib import Path
import re

import numpy as np
import trimesh

CHAINS = [['clavicle_r', 'upperarm_r', 'lowerarm_r', 'hand_r']] + [
    ['hand_r'] + [f'{f}_0{k}_r' for k in (1, 2, 3)] for f in ('thumb', 'index', 'middle', 'ring', 'pinky')]


def parse(path):
    header, frames, bone_names = [], [], None
    for line in open(path):
        if not line[:1].isdigit():
            header.append(line.rstrip('\n'))
            continue
        f = line.strip().split('|')
        i, n = 11, int(f[10])
        names, values = [], []
        for _ in range(n):
            names.append(f[i])
            values.extend(float(x) for x in f[i + 1:i + 8])
            i += 8
        bone_names = bone_names or names
        if names != bone_names:
            raise ValueError('Bone order changes between frames')
        objects = int(f[i])
        obj = [float(x) for x in f[i + 2:i + 12]] if objects else None  # loc, quat, scale
        frames.append(dict(t=float(f[2]), n=int(f[1]), head=[float(x) for x in f[3:10]], b=values, o=obj))
    return header, bone_names, frames


def decimate(mesh, cell):
    keys = np.floor(mesh.vertices / cell).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    count = inverse.max() + 1
    verts = np.zeros((count, 3))
    np.add.at(verts, inverse, mesh.vertices)
    verts /= np.bincount(inverse, minlength=count)[:, None]
    faces = inverse[mesh.faces]
    faces = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])]
    faces = faces[np.sort(np.unique(np.sort(faces, axis=1), axis=0, return_index=True)[1])]
    return verts.astype(np.float32), faces.astype(np.uint32)


def mano_payload(fit_dir, sidecar, frame_numbers, frames, hand_bone):
    world = json.loads(sidecar.read_text())['world']
    A = np.array([world['forward_in_zed_left'], world['right_in_zed_left'], world['up_in_zed_left']])
    origin = np.array(world['origin_in_zed_left_m'])
    fit = np.load(fit_dir / 'fused_hand.npz', allow_pickle=True)
    rows = {int(f): r for r, f in enumerate(fit['frame_index'])}
    picked = [rows[n] for n in frame_numbers]
    if not fit['valid'][picked].all():
        raise ValueError('Mocap frame without a valid MANO fit')
    to_world = lambda p: 100 * (p - origin) @ A.T
    verts = to_world(fit['vertices'][picked])
    wrist = to_world(fit['joints'][picked, 0])
    file_wrist = np.array([f['b'][7 * hand_bone:7 * hand_bone + 3] for f in frames])
    mismatch = float(np.abs(file_wrist - wrist).max())
    if mismatch > .01:
        raise ValueError(f'MANO wrist differs from hand_r by {mismatch:.3f} cm; wrong fit or sidecar')
    rel = np.round((verts - wrist[:, None]) * 100)  # 0.1 mm units relative to hand_r
    if np.abs(rel).max() > 32767:
        raise ValueError('Hand vertex too far from the wrist for int16')
    rel = rel.astype(np.int16)
    delta = np.concatenate([rel[:1], np.diff(rel, axis=0)])  # int16 wrap-around is undone by the cumulative sum
    packed = gzip.compress(delta.tobytes(), 9)
    faces = np.load(fit_dir / 'mano_faces.npy').astype(np.uint32)
    return dict(count=int(verts.shape[1]), gz=base64.b64encode(packed).decode(), faces=base64.b64encode(faces.tobytes()).decode(),
                wrist_mismatch_cm=mismatch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mocap', type=Path)
    parser.add_argument('--mesh', type=Path, help='Object STL (metres); omit for a take without an object')
    parser.add_argument('--output', type=Path, required=True, help='.html path; must not exist')
    parser.add_argument('--cell-mm', type=float, default=3.)
    parser.add_argument('--title', default='Mustard mocap viewer')
    parser.add_argument('--fit', type=Path, help='MANO fit directory (fused_hand.npz, mano_faces.npy) to add the hand mesh')
    parser.add_argument('--sidecar', type=Path, help='export_mocap.py .json (default: next to the Mocap file)')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists')
    header, names, frames = parse(args.mocap)
    head_text = '\n'.join(header)
    camera = re.search(r'camera:\{loc:\[([^\]]*)\],rot:\[([^\]]*)\],fov:([\d.]+),fov_h:([\d.]+),fov_v:([\d.]+)', head_text)
    found = re.search(r'mesh:([^,]+),', head_text)
    mesh_name = found.group(1) if found else 'none'
    has_object = any(f['o'] is not None for f in frames)
    if args.mesh:
        verts, faces = decimate(trimesh.load(args.mesh), args.cell_mm / 1000)
    elif has_object:  # pose without a mesh: 4 cm cube proxy at the exported object origin
        box = trimesh.creation.box(extents=(.04, .04, .04))
        verts, faces = box.vertices.astype(np.float32), box.faces.astype(np.uint32)
    else:
        verts, faces = np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint32)
    index = {n: i for i, n in enumerate(names)}
    data = dict(
        bones=names, chains=[[index[n] for n in c] for c in CHAINS], mesh=mesh_name, source=args.mocap.name,
        camera=dict(loc=[float(x) for x in camera.group(1).split(',')], rot=[float(x) for x in camera.group(2).split(',')],
                    fov_h=float(camera.group(4)), fov_v=float(camera.group(5))),
        t=[round(f['t'], 4) for f in frames], n=[f['n'] for f in frames],
        b=[[round(v, 3) for v in f['b']] for f in frames],
        o=[None if f['o'] is None else [round(v, 4) for v in f['o']] for f in frames],
        verts=base64.b64encode(verts.tobytes()).decode(), faces=base64.b64encode(faces.tobytes()).decode())
    if args.fit:
        data['mano'] = mano_payload(args.fit, args.sidecar or args.mocap.with_suffix('.json'), data['n'], frames, index['hand_r'])
    html = TEMPLATE.replace('__TITLE__', args.title).replace('__DATA__', json.dumps(data, separators=(',', ':')))
    args.output.write_text(html, encoding='utf-8')
    print(json.dumps(dict(output=str(args.output), frames=len(frames), bones=len(names), mesh_vertices=len(verts),
                          mesh_faces=len(faces), size_mb=round(len(html) / 1e6, 2)), indent=2))


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root { --bg:#f4f4f1; --panel:#ffffff; --ink:#1d1f22; --muted:#666b73; --line:#d9dad5; --accent:#2f6fde; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#141518; --panel:#1d1f23; --ink:#e9eaec; --muted:#9aa0a8; --line:#33363c; --accent:#6a9cff; } }
:root[data-theme="dark"] { --bg:#141518; --panel:#1d1f23; --ink:#e9eaec; --muted:#9aa0a8; --line:#33363c; --accent:#6a9cff; }
* { box-sizing:border-box; }
html, body { margin:0; height:100%; background:var(--bg); color:var(--ink); font:14px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif; }
#app { display:flex; flex-direction:column; height:100%; }
#view { flex:1; min-height:0; position:relative; }
canvas { display:block; }
#hud { position:absolute; left:12px; top:10px; background:color-mix(in srgb, var(--panel) 88%, transparent); border:1px solid var(--line); border-radius:8px; padding:8px 10px; font-variant-numeric:tabular-nums; }
#hud b { font-weight:600; }
#legend { position:absolute; right:12px; top:10px; background:color-mix(in srgb, var(--panel) 88%, transparent); border:1px solid var(--line); border-radius:8px; padding:8px 10px; font-size:12px; color:var(--muted); }
#legend span { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:6px; vertical-align:-1px; }
#bar { display:flex; flex-wrap:wrap; gap:10px; align-items:center; padding:10px 16px; background:var(--panel); border-top:1px solid var(--line); }
#bar button { background:var(--accent); color:#fff; border:0; border-radius:6px; padding:6px 14px; font:inherit; cursor:pointer; min-width:72px; }
#bar input[type=range] { flex:1; min-width:160px; accent-color:var(--accent); }
#bar label { color:var(--muted); white-space:nowrap; user-select:none; }
select { font:inherit; background:var(--bg); color:var(--ink); border:1px solid var(--line); border-radius:6px; padding:3px 6px; }
</style>
</head>
<body>
<div id="app">
  <div id="view">
    <div id="hud"></div>
    <div id="legend">
      <div><span style="background:#d29a76"></span>arm mesh (procedural) / MANO hand</div>
      <div><span style="background:#e0703a"></span>arm skeleton</div>
      <div><span style="background:#2f9e6e"></span>finger skeleton</div>
      <div><span style="background:#e6b800"></span>object mesh</div>
      <div><span style="background:#8a8f98"></span>ZED-left camera</div>
      <div style="margin-top:4px">bone axes: <span style="background:#e04848"></span>X <span style="background:#3aa860"></span>Y <span style="background:#3d74e0"></span>Z</div>
      <div style="margin-top:4px">drag to orbit, scroll to zoom</div>
    </div>
  </div>
  <div id="bar">
    <button id="play">Pause</button>
    <input id="scrub" type="range" min="0" value="0" step="1" aria-label="Frame">
    <label>speed <select id="speed"><option>0.25</option><option>0.5</option><option selected>1</option><option>2</option></select></label>
    <label><input id="handmesh" type="checkbox" checked> hand mesh</label>
    <label><input id="armmesh" type="checkbox" checked> arm mesh</label>
    <label><input id="skel" type="checkbox"> skeleton</label>
    <label><input id="axes" type="checkbox"> bone axes</label>
    <label><input id="obj" type="checkbox" checked> object</label>
    <label><input id="cam" type="checkbox" checked> camera</label>
    <label><input id="follow" type="checkbox"> follow hand</label>
  </div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
const D = __DATA__;
// Unreal (x fwd, y right, z up; left-handed) -> three.js (right-handed, y up): swap y and z.
const P = v => new THREE.Vector3(v[0], v[2], v[1]);
function quatCols(q) {  // columns of the rotation matrix of quaternion (x, y, z, w), in Unreal coordinates
  const [x, y, z, w] = q;
  return [[1-2*(y*y+z*z), 2*(x*y+z*w), 2*(x*z-y*w)], [2*(x*y-z*w), 1-2*(x*x+z*z), 2*(y*z+x*w)], [2*(x*z+y*w), 2*(y*z-x*w), 1-2*(x*x+y*y)]];
}
const css = getComputedStyle(document.documentElement);
const view = document.getElementById('view');
const renderer = new THREE.WebGLRenderer({antialias: true});
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
view.appendChild(renderer.domElement);
const scene = new THREE.Scene();
const camera3 = new THREE.PerspectiveCamera(45, 1, 1, 5000);
const controls = new THREE.OrbitControls(camera3, renderer.domElement);
scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 0.9));
const sun = new THREE.DirectionalLight(0xffffff, 0.7); sun.position.set(80, 200, -60); scene.add(sun);
const grid = new THREE.GridHelper(200, 20, 0x888888, 0x888888); grid.material.opacity = 0.35; grid.material.transparent = true; scene.add(grid);
scene.add(new THREE.AxesHelper(15));  // red: world X fwd, green: Z up (three Y), blue: world Y right (three Z)

function setBg() { const c = getComputedStyle(document.documentElement).getPropertyValue('--bg').trim(); scene.background = new THREE.Color(c); }
setBg(); matchMedia('(prefers-color-scheme: dark)').addEventListener('change', setBg);

// Skeleton: chain lines, joint spheres, per-bone axes, distal stubs.
const nb = D.bones.length;
const jointGeo = new THREE.SphereGeometry(0.55, 12, 8);
const joints = D.bones.map((name, i) => {
  const arm = i < 3 || name === 'clavicle_r';
  const m = new THREE.Mesh(arm ? new THREE.SphereGeometry(1.4, 16, 10) : jointGeo, new THREE.MeshStandardMaterial({color: arm ? 0xe0703a : 0x2f9e6e}));
  scene.add(m); return m;
});
function lineSet(color, count, width) {
  const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(count * 6), 3));
  const l = new THREE.LineSegments(g, new THREE.LineBasicMaterial({color, linewidth: width})); scene.add(l); return l;
}
const segs = []; D.chains.forEach((c, ci) => { for (let k = 0; k + 1 < c.length; k++) segs.push([c[k], c[k + 1], ci === 0]); });
const armTubes = []; const fingerLines = lineSet(0x2f9e6e, segs.length + 5, 2);
const tubeMat = new THREE.MeshStandardMaterial({color: 0xe0703a});
segs.filter(s => s[2]).forEach(() => { const t = new THREE.Mesh(new THREE.CylinderGeometry(0.8, 0.8, 1, 12), tubeMat); scene.add(t); armTubes.push(t); });
const axisLines = [0xe04848, 0x3aa860, 0x3d74e0].map(c => lineSet(c, nb, 1));
const distal = D.bones.map((n, i) => n.endsWith('_03_r') ? i : -1).filter(i => i >= 0);

// Object mesh from the STL (metres), placed by LOC + ROT * 100 * (x, -y, z).
const verts = new Float32Array(Uint8Array.from(atob(D.verts), c => c.charCodeAt(0)).buffer);
const faces = new Uint32Array(Uint8Array.from(atob(D.faces), c => c.charCodeAt(0)).buffer);
const og = new THREE.BufferGeometry(); og.setAttribute('position', new THREE.BufferAttribute(verts, 3)); og.setIndex(new THREE.BufferAttribute(faces, 1)); og.computeVertexNormals();
const objMesh = new THREE.Mesh(og, new THREE.MeshStandardMaterial({color: 0xe6b800, roughness: 0.6, side: THREE.DoubleSide, transparent: true, opacity: 0.9}));
objMesh.matrixAutoUpdate = false; if (faces.length) scene.add(objMesh); else document.getElementById('obj').disabled = true;

// Static ZED-left camera frustum (camera X = view direction, Y right, Z up).
const camGroup = new THREE.Group(); scene.add(camGroup);
(function () {
  const c = quatCols(D.camera.rot), o = D.camera.loc, L = 12;
  const tx = Math.tan(D.camera.fov_h * Math.PI / 360) * L, ty = Math.tan(D.camera.fov_v * Math.PI / 360) * L;
  const at = (f, r, u) => P([0, 1, 2].map(k => o[k] + f * c[0][k] + r * c[1][k] + u * c[2][k]));
  const corners = [[L, -tx, ty], [L, tx, ty], [L, tx, -ty], [L, -tx, -ty]].map(v => at(...v));
  const pts = []; corners.forEach((p, i) => { pts.push(at(0, 0, 0), p, p, corners[(i + 1) % 4]); });
  camGroup.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({color: 0x8a8f98})));
})();

// MANO hand mesh: int16 deltas (0.1 mm, relative to hand_r), gzip; decoded once at start.
const skin = new THREE.MeshStandardMaterial({color: 0xd29a76, roughness: 0.75, side: THREE.DoubleSide});
let manoRel = null, handMesh = null;
const handBone = D.bones.indexOf('hand_r');
async function decodeMano() {
  const bytes = Uint8Array.from(atob(D.mano.gz), c => c.charCodeAt(0));
  const buf = await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
  const d = new Int16Array(buf), stride = D.mano.count * 3;
  for (let i = stride; i < d.length; i++) d[i] += d[i - stride];  // undo frame-to-frame deltas (int16 wraps back exactly)
  manoRel = d;
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(stride), 3));
  g.setIndex(new THREE.BufferAttribute(new Uint32Array(Uint8Array.from(atob(D.mano.faces), c => c.charCodeAt(0)).buffer), 1));
  handMesh = new THREE.Mesh(g, skin); scene.add(handMesh);
}
function updateHand(i, b) {
  if (!handMesh) return;
  const a = handMesh.geometry.attributes.position, n = D.mano.count, base = i * n * 3, w = b.slice(handBone * 7, handBone * 7 + 3);
  for (let k = 0; k < n; k++) {
    const o = base + 3 * k;  // Unreal (x, y, z) -> three.js (x, z, y)
    a.setXYZ(k, w[0] + manoRel[o] / 100, w[2] + manoRel[o + 2] / 100, w[1] + manoRel[o + 1] / 100);
  }
  a.needsUpdate = true; handMesh.geometry.computeVertexNormals(); handMesh.geometry.computeBoundingSphere();
  handMesh.visible = document.getElementById('handmesh').checked;
}

// Procedural arm: tapered limbs whose cross-section axes come from the bone frames.
const RINGS = 24, SIDES = 28;
function makeLimb() {
  const g = new THREE.BufferGeometry(), idx = [];
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(RINGS * SIDES * 3), 3));
  for (let r = 0; r + 1 < RINGS; r++) for (let k = 0; k < SIDES; k++) {
    const a = r * SIDES + k, b = r * SIDES + (k + 1) % SIDES, c = a + SIDES, d = b + SIDES;
    idx.push(a, c, b, b, c, d);
  }
  g.setIndex(idx); const m = new THREE.Mesh(g, skin); scene.add(m); return m;
}
function interp(table, s) {  // table: [[s, ru, rv], ...] sorted by s; smoothstep between rows
  for (let k = 0; k + 1 < table.length; k++) if (s <= table[k + 1][0]) {
    const [s0, a0, b0] = table[k], [s1, a1, b1] = table[k + 1], t = (s - s0) / (s1 - s0), e = t * t * (3 - 2 * t);
    return [a0 + (a1 - a0) * e, b0 + (b1 - b0) * e];
  }
  return table[table.length - 1].slice(1);
}
function updateLimb(m, start, end, axis, table) {
  const d = end.clone().sub(start), dn = d.clone().normalize();
  const u = axis.clone().addScaledVector(dn, -axis.dot(dn)).normalize(), v = new THREE.Vector3().crossVectors(dn, u);
  const a = m.geometry.attributes.position;
  for (let r = 0; r < RINGS; r++) {
    const s = r / (RINGS - 1), [ru, rv] = interp(table, s), c = start.clone().addScaledVector(d, s);
    for (let k = 0; k < SIDES; k++) {
      const t = 2 * Math.PI * k / SIDES, p = c.clone().addScaledVector(u, ru * Math.cos(t)).addScaledVector(v, rv * Math.sin(t));
      a.setXYZ(r * SIDES + k, p.x, p.y, p.z);
    }
  }
  a.needsUpdate = true; m.geometry.computeVertexNormals(); m.geometry.computeBoundingSphere();
}
// Radii in cm: [fraction along bone, radius along u, radius along v].
const CLAVICLE = [[0, 2.6, 2.6], [1, 4.2, 4.2]];
const UPPER = [[0, 4.9, 4.9], [0.3, 5.0, 4.7], [0.75, 4.3, 4.1], [1, 3.9, 3.9]];
const FORE = [[0, 3.9, 3.9], [0.25, 4.2, 4.0], [0.7, 3.2, 2.5], [1, 2.8, 1.9]];  // u = hand lateral (wrist width), v = palm normal
const limbs = {clav: makeLimb(), upper: makeLimb(), fore: makeLimb()};
const shoulderBall = new THREE.Mesh(new THREE.SphereGeometry(4.9, 28, 18), skin), elbowBall = new THREE.Mesh(new THREE.SphereGeometry(3.9, 24, 16), skin);
scene.add(shoulderBall, elbowBall);
function updateArm(pos, cols) {
  const on = document.getElementById('armmesh').checked;
  [limbs.clav, limbs.upper, limbs.fore, shoulderBall, elbowBall].forEach(m => m.visible = on);
  if (!on) return;
  const [cl, ua, la, hd] = ['clavicle_r', 'upperarm_r', 'lowerarm_r', 'hand_r'].map(n => D.bones.indexOf(n));
  updateLimb(limbs.clav, pos[cl], pos[ua], P(cols[cl][2]), CLAVICLE);
  updateLimb(limbs.upper, pos[ua], pos[la], P(cols[ua][2]), UPPER);
  updateLimb(limbs.fore, pos[la], pos[hd], P(cols[hd][2]), FORE);
  shoulderBall.position.copy(pos[ua]); elbowBall.position.copy(pos[la]);
}

let cur = 0;
const up = new THREE.Vector3(0, 1, 0);
function show(i) {
  cur = i; const b = D.b[i];
  const pos = [], cols = [];
  for (let k = 0; k < nb; k++) { const o = k * 7; pos.push(P(b.slice(o, o + 3))); cols.push(quatCols(b.slice(o + 3, o + 7))); joints[k].position.copy(pos[k]); }
  const fl = fingerLines.geometry.attributes.position; let fi = 0, ai = 0;
  segs.forEach(([a, c, arm]) => {
    if (arm) {
      const t = armTubes[ai++], d = pos[c].clone().sub(pos[a]);
      t.position.copy(pos[a]).addScaledVector(d, 0.5); t.scale.set(1, d.length(), 1); t.quaternion.setFromUnitVectors(up, d.normalize());
    } else { fl.setXYZ(fi++, pos[a].x, pos[a].y, pos[a].z); fl.setXYZ(fi++, pos[c].x, pos[c].y, pos[c].z); }
  });
  distal.forEach(k => {  // fingertip stub: 2 cm along the distal bone's -X
    const e = pos[k].clone().add(P(cols[k][0].map(v => -2 * v)));
    fl.setXYZ(fi++, pos[k].x, pos[k].y, pos[k].z); fl.setXYZ(fi++, e.x, e.y, e.z);
  });
  fl.needsUpdate = true;
  const sk = document.getElementById('skel').checked;
  joints.forEach(j => j.visible = sk); armTubes.forEach(t => t.visible = sk); fingerLines.visible = sk;
  updateArm(pos, cols); updateHand(i, b);
  axisLines.forEach((l, ax) => {
    const a = l.geometry.attributes.position;
    for (let k = 0; k < nb; k++) {
      const len = k < 4 ? 6 : 2, e = pos[k].clone().add(P(cols[k][ax].map(v => v * len)));
      a.setXYZ(2 * k, pos[k].x, pos[k].y, pos[k].z); a.setXYZ(2 * k + 1, e.x, e.y, e.z);
    }
    a.needsUpdate = true; l.visible = document.getElementById('axes').checked;
  });
  const o = D.o[i];
  objMesh.visible = !!o && document.getElementById('obj').checked;
  if (o) {
    const c = quatCols(o.slice(3, 7)), s = 100, k = o.length > 7 ? o.slice(7, 10) : [1, 1, 1];
    // world_three = swap(LOC + R * diag(1,-1,1) * 100 v); rows are three.js x, y, z = Unreal x, z, y.
    const r = [0, 2, 1];
    const m = new THREE.Matrix4().set(
      s * k[0] * c[0][r[0]], -s * k[1] * c[1][r[0]], s * k[2] * c[2][r[0]], o[r[0]],
      s * k[0] * c[0][r[1]], -s * k[1] * c[1][r[1]], s * k[2] * c[2][r[1]], o[r[1]],
      s * k[0] * c[0][r[2]], -s * k[1] * c[1][r[2]], s * k[2] * c[2][r[2]], o[r[2]],
      0, 0, 0, 1);
    objMesh.matrix.copy(m);
  }
  if (document.getElementById('follow').checked) { const h = pos[D.bones.indexOf('hand_r')]; controls.target.lerp(h, 0.2); }
  const gap = i > 0 && D.n[i] - D.n[i - 1] > 1 ? ` &nbsp;(gap: ${D.n[i] - D.n[i - 1] - 1} frames skipped)` : '';
  document.getElementById('hud').innerHTML = `<b>${D.source}</b><br>frame ${D.n[i]} &nbsp; ${D.t[i].toFixed(2)} s &nbsp; ${i + 1}/${D.t.length}<br>object: ${o ? D.mesh : 'no pose'}${gap}`;
  scrub.value = i;
}

const scrub = document.getElementById('scrub'); scrub.max = D.t.length - 1;
scrub.addEventListener('input', () => { playing = false; playBtn.textContent = 'Play'; show(+scrub.value); });
const playBtn = document.getElementById('play'); let playing = true;
playBtn.addEventListener('click', () => { playing = !playing; playBtn.textContent = playing ? 'Pause' : 'Play'; clock = performance.now(); simT = D.t[cur]; });
['axes', 'obj', 'cam', 'handmesh', 'armmesh', 'skel'].forEach(id => document.getElementById(id).addEventListener('change', () => { camGroup.visible = document.getElementById('cam').checked; show(cur); }));

// Initial view: from behind-left of the subject, looking at the first hand position.
const h0 = P(D.b[0].slice(3 * 7, 3 * 7 + 3));
controls.target.copy(h0); camera3.position.set(h0.x - 90, h0.y + 70, h0.z - 70); controls.update();

function resize() { const w = view.clientWidth, h = view.clientHeight; renderer.setSize(w, h); camera3.aspect = w / h; camera3.updateProjectionMatrix(); }
addEventListener('resize', resize); resize();
let clock = performance.now(), simT = D.t[0];
function tick(now) {
  if (playing) {
    simT += (now - clock) / 1000 * +document.getElementById('speed').value;
    if (simT > D.t[D.t.length - 1]) simT = D.t[0], cur = 0;
    let i = cur; while (i + 1 < D.t.length && D.t[i + 1] <= simT) i++;
    if (i !== cur || i === 0) show(i);
  }
  clock = now; controls.update(); renderer.render(scene, camera3); requestAnimationFrame(tick);
}
(async () => {
  if (D.mano) {
    document.getElementById('hud').textContent = 'decoding hand mesh...';
    try { await decodeMano(); } catch (e) { document.getElementById('handmesh').disabled = true; console.error(e); }
  } else document.getElementById('handmesh').disabled = true;
  const m = location.hash.match(/i=(\d+)/);  // e.g. viewer.html#i=1200 opens paused at that row
  if (m) { playing = false; playBtn.textContent = 'Play'; show(Math.min(+m[1], D.t.length - 1)); simT = D.t[cur]; } else show(0);
  if (m) { const h = P(D.b[cur].slice(handBone * 7, handBone * 7 + 3)); controls.target.copy(h); camera3.position.set(h.x - 45, h.y + 30, h.z - 35); }
  requestAnimationFrame(tick);
})();
</script>
</body>
</html>
"""

if __name__ == '__main__':
    main()
