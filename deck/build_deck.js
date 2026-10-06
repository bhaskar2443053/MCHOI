// Build docs/MCHOI_v1.0_pipeline.pptx: dataset summary and step-by-step workflow with real snapshots and GIFs.
// Run through deck/build.py, which sets MCHOI_DATASET_DIR and DECK_OUT and injects the equations afterwards.
const pptxgen = require('pptxgenjs');
const fs = require('fs');
const path = require('path');
const { applyTheme } = require('./theme');
// image-size comes with pptxgenjs.
const { imageSize } = require(require.resolve('image-size', { paths: [require.resolve('pptxgenjs')] }));

if (!process.env.MCHOI_DATASET_DIR) throw new Error('Set MCHOI_DATASET_DIR (or run python deck/build.py)');
const ROOT = path.resolve(process.env.MCHOI_DATASET_DIR);
const MEDIA = path.join(ROOT, 'docs', 'media');
const OUT = process.env.DECK_OUT || path.join(ROOT, 'docs', 'MCHOI_v1.0_pipeline.pptx');

const THEME = {
  name: 'MCHOI', headFontFace: 'Cambria', bodyFontFace: 'Calibri',
  colors: {
    dk1: '1B1F27', lt1: 'FFFFFF', dk2: '14303A', lt2: 'EEF2F3',
    accent1: '0E7C86', accent2: 'E8913A', accent3: 'C2463B', accent4: '5B6C8F', accent5: '7FB7BE', accent6: 'D9B44A',
    hlink: '0E7C86', folHlink: '5B6C8F',
  },
};
const HEX = THEME.colors;

const pres = new pptxgen();
pres.layout = 'LAYOUT_WIDE'; // 13.33 x 7.5 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = 'MCHOI v1.0: dataset and processing workflow';
pres.author = 'MCHOI dataset';
const C = pres.SchemeColor;
const W = 13.333, H = 7.5;

pres.defineSlideMaster({
  title: 'TITLE_DARK',
  background: { color: HEX.dk2 },
  objects: [
    { placeholder: { options: { name: 'title', type: 'title', x: 0.6, y: 1.6, w: 6.6, h: 2.2, fontSize: 40, bold: true, color: C.background1, valign: 'bottom', align: 'left', fontFace: THEME.headFontFace }, text: '' } },
    { placeholder: { options: { name: 'body', type: 'body', x: 0.6, y: 3.95, w: 6.4, h: 1.9, fontSize: 18, color: 'CFE3E5', valign: 'top' }, text: '' } },
  ],
});
pres.defineSlideMaster({
  title: 'DARK_CONTENT',
  background: { color: HEX.dk2 },
  objects: [
    { placeholder: { options: { name: 'title', type: 'title', x: 0.6, y: 0.45, w: 12.1, h: 0.95, fontSize: 32, bold: true, color: C.background1, valign: 'middle', align: 'left', fontFace: THEME.headFontFace }, text: '' } },
  ],
});
pres.defineSlideMaster({
  title: 'STEP',
  background: { color: HEX.lt1 },
  margin: [0.5, 0.5, 0.6, 0.5],
  objects: [
    { placeholder: { options: { name: 'title', type: 'title', x: 1.35, y: 0.3, w: 11.4, h: 0.85, fontSize: 30, bold: true, color: C.text2, valign: 'middle', align: 'left', fontFace: THEME.headFontFace }, text: '' } },
    { text: { text: 'MCHOI v1.0 · dataset and processing workflow', options: { x: 0.55, y: 7.02, w: 8, h: 0.3, fontSize: 10, color: '6B7A80' } } },
  ],
  slideNumber: { x: 12.3, y: 7.02, w: 0.6, h: 0.3, fontSize: 10, color: '6B7A80', align: 'right' },
});
pres.defineSlideMaster({
  title: 'CONTENT',
  background: { color: HEX.lt1 },
  margin: [0.5, 0.5, 0.6, 0.5],
  objects: [
    { placeholder: { options: { name: 'title', type: 'title', x: 0.55, y: 0.3, w: 12.2, h: 0.85, fontSize: 30, bold: true, color: C.text2, valign: 'middle', align: 'left', fontFace: THEME.headFontFace }, text: '' } },
    { text: { text: 'MCHOI v1.0 · dataset and processing workflow', options: { x: 0.55, y: 7.02, w: 8, h: 0.3, fontSize: 10, color: '6B7A80' } } },
  ],
  slideNumber: { x: 12.3, y: 7.02, w: 0.6, h: 0.3, fontSize: 10, color: '6B7A80', align: 'right' },
});

// Image helpers: fit inside a box, keep aspect ratio, centre in the box.
function fit(name, x, y, w, h) {
  const { width: iw, height: ih } = imageSize(path.join(MEDIA, name));
  const s = Math.min(w / iw, h / ih);
  const fw = iw * s, fh = ih * s;
  return { path: path.join(MEDIA, name), x: x + (w - fw) / 2, y: y + (h - fh) / 2, w: fw, h: fh };
}
function picture(slide, name, x, y, w, h, caption) {
  const f = fit(name, x, y, w, h);
  slide.addImage({ ...f, objectName: name, altText: caption || name });
  if (caption) slide.addText(caption, { x: f.x, y: f.y + f.h + 0.05, w: f.w, h: 0.3, fontSize: 11, italic: true, color: '5A6870', margin: 0, isTextBox: true });
  return f;
}
// Motif: numbered step badge in a teal circle next to the title.
function badge(slide, n) {
  slide.addShape(pres.shapes.OVAL, { x: 0.55, y: 0.42, w: 0.62, h: 0.62, fill: { color: C.accent1 }, line: { color: C.accent1 }, objectName: 'step badge' });
  slide.addText(String(n), { x: 0.55, y: 0.42, w: 0.62, h: 0.62, fontSize: 20, bold: true, color: C.background1, align: 'center', valign: 'middle', margin: 0, isTextBox: true });
}
function stepTitle(slide, n, text) {
  badge(slide, n);
  slide.addText(text, { placeholder: 'title' });
}
function bullets(slide, items, x, y, w, h, size = 15) {
  slide.addText(items.map((t, i) => {
    const run = typeof t === 'string' ? { text: t, options: {} } : t;
    return { text: run.text, options: { bullet: run.options.bullet === false ? false : true, breakLine: i < items.length - 1, ...run.options } };
  }), { x, y, w, h, fontSize: size, color: C.text1, valign: 'top', paraSpaceAfter: 6, isTextBox: true });
}
function label(slide, text, x, y, w) {
  slide.addText(text, { x, y, w, h: 0.35, fontSize: 13, bold: true, color: C.accent1, margin: 0, isTextBox: true, charSpacing: 1 });
}
function card(slide, x, y, w, h, fill = HEX.lt2) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { color: fill } });
}
function note(slide, text) { slide.addNotes(text); }

// Rich text: a_b / a_{bc} subscript, a^b / a^{bc} superscript, **v** bold (vectors).
function rich(str, base = {}) {
  const runs = [];
  let buf = '', i = 0;
  const push = (t, o) => { if (t) runs.push({ text: t, options: { ...base, ...o } }); };
  while (i < str.length) {
    const ch = str[i];
    if ((ch === '_' || ch === '^') && i + 1 < str.length) {
      push(buf, {}); buf = '';
      let t;
      if (str[i + 1] === '{') { const j = str.indexOf('}', i + 2); t = str.slice(i + 2, j); i = j + 1; } else { t = str[i + 1]; i += 2; }
      push(t, ch === '_' ? { subscript: true } : { superscript: true });
    } else if (str.startsWith('**', i)) {
      push(buf, {}); buf = '';
      const j = str.indexOf('**', i + 2);
      push(str.slice(i + 2, j), { bold: true, italic: false });
      i = j + 2;
    } else { buf += ch; i++; }
  }
  push(buf, {});
  return runs;
}
const EQ = JSON.parse(fs.readFileSync(path.join(__dirname, 'eq_sizes.json'), 'utf8'));
// Equations are native PowerPoint equations: a placeholder text box here, replaced by OMML in inject_equations.py.
// Size comes from the LaTeX-rendered metrics (20 pt reference), widened/heightened for Cambria Math.
// Sizes per point come from PowerPoint itself (measure_equations.ps1 -> eq_measured.json) when available,
// otherwise from the LaTeX-rendered metrics widened/heightened for Cambria Math.
const WF = 1.12, HF = 1.3;
const MEASURED_PATH = path.join(__dirname, 'eq_measured.json');
const MEASURED = fs.existsSync(MEASURED_PATH) ? JSON.parse(fs.readFileSync(MEASURED_PATH, 'utf8')) : {};
function eqDims(name) {  // inches at 20 pt
  if (MEASURED[name]) return [MEASURED[name].w * 20, MEASURED[name].h * 20];
  const [pw, ph] = EQ.images[name];
  return [pw / EQ.dpi * WF, ph / EQ.dpi * HF];
}
function eqScale(name, maxW, maxH, cap) {
  const [w, h] = eqDims(name);
  return Math.min(cap, 0.97 * maxW / w, maxH / h);
}
function equation(slide, name, x, y, maxW, maxH, cap = 0.75) {
  const pt = Math.floor(20 * eqScale(name, maxW, maxH, cap) * 2) / 2;
  const h = eqDims(name)[1] * pt / 20;
  slide.addText(`⟦EQ:${name}:${pt}⟧`, { x, y, w: maxW, h, fontSize: pt, margin: 0, valign: 'top', wrap: false, color: HEX.dk1, isTextBox: true, objectName: 'equation ' + name });
  return { w: maxW, h };
}
// Math slide: equations with a plain-language line under each (left), symbol table (right).
function mathSlide(stage, title, blocks, rows, notes, master = 'STEP') {
  const slide = pres.addSlide({ masterName: master, sectionTitle: SEC });
  if (stage) stepTitle(slide, stage, title); else slide.addText(title, { placeholder: 'title' });
  const X = 0.55, WL = 7.1, TOP = 1.35, BOTTOM = 6.8;
  // Largest equation scale (and text size) for which the column still fits above the footer.
  const eqSize = (b, cap) => eqDims(b.eq)[1] * eqScale(b.eq, WL, (b.maxH || 0.6) * cap / 0.75, cap);
  const layout = (cap, font) => blocks.reduce((y, b) => y + eqSize(b, cap) + 0.07 + Math.ceil(b.text.length / (92 * 12.5 / font)) * font / 52 + 0.04 + (b.gap ?? 0.2), TOP);
  let cap = 0.62, font = 12.5;
  for (const [c, f] of [[1.0, 14], [0.92, 13.5], [0.85, 13], [0.78, 13], [0.72, 12.5], [0.66, 12.5], [0.62, 12.5]]) {
    if (layout(c, f) <= BOTTOM) { cap = c; font = f; break; }
  }
  let y = TOP;
  blocks.forEach(b => {
    const e = equation(slide, b.eq, X, y, WL, (b.maxH || 0.6) * cap / 0.75, cap);
    y += e.h + 0.07;
    const h = Math.ceil(b.text.length / (92 * 12.5 / font)) * font / 52 + 0.04;
    slide.addText(rich(b.text), { x: X, y, w: WL, h, fontSize: font, color: '3D4A52', margin: 0, valign: 'top', isTextBox: true });
    y += h + (b.gap ?? 0.2);
  });
  if (y > 7.0) console.warn(`WARNING: "${title}" left column ends at ${y.toFixed(2)} in`);
  label(slide, 'SYMBOLS', 7.95, 1.35, 4.8);
  const head = ['Symbol', 'Meaning', 'Value / unit'].map(t => ({ text: t, options: { bold: true, color: HEX.lt1, fill: { color: HEX.accent1 } } }));
  const body = rows.map((r, i) => r.map((c, j) => ({ text: rich(c, j === 0 ? { italic: true } : {}), options: { color: HEX.dk1, fill: { color: i % 2 ? HEX.lt2 : 'FFFFFF' } } })));
  slide.addTable([head, ...body], { x: 7.95, y: 1.72, w: 4.85, colW: [0.95, 2.65, 1.25], fontSize: 11, border: { type: 'solid', pt: 0.5, color: 'D5DDE0' }, valign: 'middle', margin: [2, 4, 2, 4] });
  if (notes) note(slide, notes);
  return slide;
}
let SEC = 'Overview';


// 1. Title
pres.addSection({ title: 'Overview' });
let s = pres.addSlide({ masterName: 'TITLE_DARK', sectionTitle: 'Overview' });
s.addText('MCHOI v1.0: Multi-Camera Hand–Object Interaction dataset', { placeholder: 'title' });
s.addText('Right-hand MANO pose, arm and 6-DoF YCB object pose from four calibrated RGB-D views. Dataset contents and the processing workflow, step by step.', { placeholder: 'body' });
s.addText('Recorded 2026-09-23 · 9 recordings · 13.3 min', { x: 0.6, y: 6.3, w: 6.4, h: 0.4, fontSize: 14, color: '9FC3C7', isTextBox: true });
picture(s, 'anim_all_recordings.gif', 7.45, 1.35, 5.4, 4.6);
note(s, 'Animated overview: the nine recordings with the exported hand mesh, arm and object mesh drawn over the ZED left camera.');

// 2. At a glance
s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Overview' });
s.addText('Dataset at a glance', { placeholder: 'title' });
const stats = [['9', 'recordings', '7 object runs, 2 gesture runs'], ['13.3 min', 'total duration', '795.8 s at ~30 fps'], ['22,887', 'synchronized frames', 'ZED reference clock'], ['71 %', 'frames with a hand pose', '16,263 valid MANO fits']];
stats.forEach(([big, small, sub], i) => {
  const x = 0.55 + i * 3.1;
  card(s, x, 1.45, 2.85, 2.1);
  s.addText(big, { x: x + 0.2, y: 1.6, w: 2.5, h: 0.95, fontSize: 40, bold: true, color: C.accent1, fontFace: THEME.headFontFace, margin: 0, isTextBox: true });
  s.addText(small, { x: x + 0.2, y: 2.55, w: 2.5, h: 0.4, fontSize: 15, bold: true, color: C.text2, margin: 0, isTextBox: true });
  s.addText(sub, { x: x + 0.2, y: 2.95, w: 2.5, h: 0.4, fontSize: 12, color: '5A6870', margin: 0, isTextBox: true });
});
label(s, 'OBJECTS (YCB MESHES INCLUDED)', 0.55, 4.05, 6);
bullets(s, ['Mustard bottle (006), cracker box (003, party size, fitted scale), pudding box (008), bowl (024), mug (025, two runs)',
  'Banana-shaped object: pose only (no matching mesh)', 'Free gestures without an object (two runs)'], 0.55, 4.45, 6.1, 2.4, 16);
label(s, 'PER RECORDING', 6.95, 4.05, 6);
bullets(s, ['RGB video of 4 views + depth of 3 (ZED SDK, 2 × RealSense)', 'MANO hand pose, right-arm joints, object pose',
  'Motion-capture file (Unreal skeleton), 3D viewer, overlay video', 'Rig calibration with all 25 ChArUco captures'], 6.95, 4.45, 5.9, 2.4, 16);
note(s, 'Numbers from recordings.tsv. Frames without a hand pose are marked invalid, never interpolated.');

// 3. Capture setup
s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Overview' });
s.addText('Capture setup: four calibrated views', { placeholder: 'title' });
picture(s, 'step02_raw_four_views.jpg', 0.55, 1.35, 7.6, 4.3, 'Same moment in all four views (mustard bottle, ZED frame 1530).');
const camRows = [['Camera', 'Device', 'Streams'], ['zedminileft', 'ZED Mini, left lens', 'RGB 1080p; SDK depth'], ['zedminiright', 'ZED Mini, right lens', 'RGB 1080p'],
  ['d435n1', 'RealSense D435 #1 (side)', 'RGB 1080p; depth 848×480'], ['d435n2', 'RealSense D435 #2 (facing)', 'RGB 1080p; depth 848×480']];
s.addTable(camRows.map((r, i) => r.map(c => ({ text: c, options: { bold: i === 0, color: i === 0 ? HEX.lt1 : HEX.dk1, fill: { color: i === 0 ? HEX.accent1 : (i % 2 ? 'FFFFFF' : HEX.lt2) } } }))),
  { x: 8.45, y: 1.4, w: 4.4, colW: [1.25, 1.75, 1.4], fontSize: 11, border: { type: 'solid', pt: 0.5, color: 'D5DDE0' }, valign: 'middle' });
bullets(s, ['ZED-left optical frame is the reference for all 3D results', 'Cameras run freely at ~30 fps; no hardware trigger',
  'Native SVO2/DB3 files (~320 GB) converted to standard video'], 8.45, 3.85, 4.4, 2.4, 14);

// 4. Workflow overview (five stages)
SEC = 'Workflow';
pres.addSection({ title: SEC });
s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: SEC });
s.addText('Processing workflow: five stages', { placeholder: 'title' });
s.addText([{ text: 'Inputs  ', options: { bold: true, color: HEX.accent1 } }, { text: 'four camera recordings (ZED Mini stereo, 2 × RealSense D435) · ChArUco board images · YCB object meshes' }],
  { x: 0.55, y: 1.25, w: 12.2, h: 0.4, fontSize: 14, color: C.text1, margin: 0, isTextBox: true });
const STAGES = [
  ['Calibrate', ['ChArUco board, 25 poses', 'lens model per camera', 'camera poses, ZED frame'], 'K_c, D_c, T_{c←z}'],
  ['Synchronize & export', ['RealSense clock offsets', 'frames paired ≤ 20 ms', 'RGB + lossless depth'], 'Δ_c, paired frames'],
  ['Detect & track', ['ArUco marker poses', 'smoothed hand boxes', 'WiLoR MANO per view'], 'T_{z←m}(t), 2D keypoints'],
  ['Fit hand, arm & object', ['one MANO hand, all views', 'shoulder and elbow', 'object mesh on marker'], 'θ_t, **s**, τ_t, T_{z←o}(t)'],
  ['Export & visualize', ['gravity-aligned frame', 'mocap file (Unreal)', 'viewer + overlay video'], 'mocap file, viewer, videos'],
];
const CWD = 2.25, GAP = 0.24, CY = 1.85, CHT = 4.0;
STAGES.forEach(([name, does, out], i) => {
  const x = 0.55 + i * (CWD + GAP);
  card(s, x, CY, CWD, CHT, i % 2 ? HEX.lt2 : 'E3F1F2');
  s.addShape(pres.shapes.OVAL, { x: x + 0.18, y: CY + 0.2, w: 0.62, h: 0.62, fill: { color: C.accent1 }, line: { color: C.accent1 } });
  s.addText(String(i + 1), { x: x + 0.18, y: CY + 0.2, w: 0.62, h: 0.62, fontSize: 20, bold: true, color: C.background1, align: 'center', valign: 'middle', margin: 0, isTextBox: true });
  s.addText(name, { x: x + 0.18, y: CY + 0.95, w: CWD - 0.3, h: 0.7, fontSize: 17, bold: true, color: C.text2, valign: 'top', margin: 0, isTextBox: true, fontFace: THEME.headFontFace });
  bullets(s, does, x + 0.12, CY + 1.65, CWD - 0.2, 1.5, 12.5);
  s.addShape(pres.shapes.LINE, { x: x + 0.18, y: CY + 3.15, w: CWD - 0.36, h: 0, line: { color: 'C9D6D9', width: 1 } });
  s.addText([{ text: 'Output  ', options: { bold: true, color: HEX.accent1, breakLine: true } }, ...rich(out, { italic: true })],
    { x: x + 0.18, y: CY + 3.22, w: CWD - 0.3, h: 0.7, fontSize: 12.5, color: C.text1, valign: 'top', margin: 0, isTextBox: true });
  if (i < 4) s.addShape(pres.shapes.RIGHT_ARROW, { x: x + CWD + 0.02, y: CY + 1.8, w: 0.2, h: 0.36, fill: { color: C.accent2 }, line: { color: C.accent2 } });
});
s.addText([{ text: 'Outputs  ', options: { bold: true, color: HEX.accent1 } }, { text: 'RGB-D videos · MANO hand pose · right-arm joints · object pose · mocap file · 3D viewer · overlay video' }],
  { x: 0.55, y: 6.1, w: 12.2, h: 0.4, fontSize: 14, color: C.text1, margin: 0, isTextBox: true });
s.addText('Stage 1 runs once per rig set-up; stages 2–5 run per recording (python -m mchoi.pipeline).', { x: 0.55, y: 6.5, w: 12.2, h: 0.35, fontSize: 12, italic: true, color: '5A6870', margin: 0, isTextBox: true });
note(s, 'Each stage has a math slide (equations, what they compute, symbol table) followed by real example snapshots.');

// Notation
mathSlide(0, 'Notation and conventions', [
  { eq: 'n1', text: 'Rigid transform T_{a←b}: rotation R and translation **t** that map a point from frame b into frame a; transforms chain by multiplication.' },
  { eq: 'n2', text: 'Projection π: pinhole camera with lens distortion; gives the pixel at which a 3D point **X** = (X, Y, Z), expressed in the camera frame, appears.' },
  { eq: 'n3', text: 'Brown lens distortion d_D: radial (k_1, k_2) and tangential (p_1, p_2) terms, k_3 = 0; r^2 = x^2 + y^2 on the normalised image plane.' },
  { eq: 'n4', text: 'Rotation maps: Exp turns an axis–angle vector ω (radians) into a rotation matrix; Log is its inverse. Used to filter and interpolate rotations.' },
], [
  ['z', 'ZED-left optical frame: x right, y down, z forward (reference)', 'metres'],
  ['c', 'camera: zedminileft, zedminiright, d435n1, d435n2', '—'],
  ['b, m, o', 'ChArUco board, ArUco marker, object (mesh) frame', '—'],
  ['w', 'world frame (Unreal): X forward, Y right, Z up', 'cm'],
  ['T_{a←b}', 'rigid transform (R_{a←b}, **t**_{a←b})', 'SE(3)'],
  ['K', 'intrinsics f_x, f_y, c_x, c_y', 'px'],
  ['D', 'distortion k_1, k_2, p_1, p_2', '—'],
  ['t, n', 'time; frame index (ZED clock)', 's; —'],
  ['OE(·)', 'One Euro filter (stage 3)', '—'],
  ['ρ_s(·)', 'Huber (smooth-L1) loss at scale s', '—'],
  ['**x**, x̂', 'bold: vector; hat: estimate or filtered value', '—'],
], 'Notation used on the following math slides. All 3D results are expressed in frame z unless stated otherwise.', 'CONTENT');


SEC = '1 · Calibrate';
pres.addSection({ title: SEC });
mathSlide(1, 'Calibrate: camera model and objective', [
  { eq: 'c1', text: 'Model: where corner k of the board in capture i should appear in camera c.' },
  { eq: 'c2', maxH: 0.75, text: 'Fit: the intrinsics, distortion, camera poses and board poses that best explain every detected corner. Solved in stages: per-camera intrinsics, pairwise stereo with fixed intrinsics, then camera poses chained over a graph to the ZED-left reference (T_{z←z} = I).' },
  { eq: 'c3', maxH: 0.75, text: 'Quality: reprojection error per camera: 0.31, 0.34, 0.22 and 0.35 px for this rig.' },
], [
  ['**u**_{cik}', 'detected pixel of corner k, capture i, camera c', 'px'],
  ['**X**^{b}_{k}', 'corner k on the board', '30 mm grid'],
  ['K_c', 'intrinsics of camera c', 'px'],
  ['D_c', 'lens distortion of camera c', '—'],
  ['T_{c←z}', 'pose of the ZED-left frame in camera c (extrinsics)', 'm'],
  ['T_{z←b}', 'board pose (one per capture i)', 'm'],
  ['i', 'board capture', '1 … 25'],
  ['N_c', 'corners used for camera c', '—'],
], 'Calibration app: per-view Zhang calibration, pairwise stereo with fixed intrinsics, then graph composition to the ZED-left frame. The capture checks (≥ 8 corners per view, host-time spread < 80 ms, board still) are on the next slide.');


// 5. Calibration (capture)
s = pres.addSlide({ masterName: 'STEP', sectionTitle: SEC });
stepTitle(s, 1, 'Rig calibration: ChArUco capture');
picture(s, 'step01_calibration_charuco.jpg', 0.55, 1.35, 7.7, 4.35, 'One of the 25 board poses; green = detected ChArUco corners.');
label(s, 'WORKFLOW', 8.55, 1.4, 4);
bullets(s, ['Board: 5 × 7 squares, 30 mm squares, 22 mm markers, DICT_4X4_50, flat rigid backing',
  'Cameras fixed; board moved through positions, distances and tilts; held ≥ 0.6 s until all views show "Ready"',
  'Capture taken only if every view sees ≥ 8 corners and host-time spread < 80 ms',
  'Images saved lossless at native resolution with exact corner IDs (calibration/captures/)'], 8.55, 1.8, 4.3, 4.6, 14);
note(s, 'Calibration app: Start Calibration.cmd (calibration/ package). It was run after the recordings, with the rig untouched.');

// 6. Calibration (solve)
s = pres.addSlide({ masterName: 'STEP', sectionTitle: SEC });
stepTitle(s, 1, 'Rig calibration: solve and verify');
picture(s, 'step01_calibration_coverage.jpg', 0.55, 1.35, 7.7, 4.35, 'Image area covered by the accepted board poses, per camera.');
label(s, 'SOLVE', 8.55, 1.4, 4);
bullets(s, ['Per-view intrinsics + Brown distortion (k3 = 0)', 'Pairwise stereo with fixed intrinsics; connected graph → transforms to the ZED-left frame',
  'Check: alternate graph paths agree; click-to-verify in all views'], 8.55, 1.8, 4.3, 2.3, 14);
card(s, 8.55, 4.25, 4.3, 1.55);
s.addText([{ text: 'Reprojection RMS (px)', options: { bold: true, color: HEX.dk2, breakLine: true } },
  { text: 'zedminileft 0.31 · zedminiright 0.34', options: { breakLine: true } }, { text: 'd435n1 0.22 · d435n2 0.35' }],
  { x: 8.7, y: 4.35, w: 4.0, h: 1.35, fontSize: 14, color: C.text1, valign: 'middle', isTextBox: true });
s.addText('Dataset: calibration/calib-01_rig_calibration.json (.yml) + 125 capture files', { x: 8.55, y: 5.95, w: 4.3, h: 0.6, fontSize: 12, italic: true, color: '5A6870', isTextBox: true });

SEC = '2 · Synchronize & export';
pres.addSection({ title: SEC });
mathSlide(2, 'Synchronize & export: clock offsets, pairing, depth', [
  { eq: 'y1', maxH: 0.7, text: 'Clock offset: the time shift that makes the same ArUco corners agree between a RealSense camera and the ZED (distance to the epipolar line). Fitted on alternating 2 s blocks; accepted only if it also holds on the held-out blocks.' },
  { eq: 'y2', text: 'Pairing: each ZED frame n takes the nearest unused frame j of camera c once the offset is removed; nothing is paired beyond 20 ms.' },
  { eq: 'y3', text: 'Depth: a stored depth value becomes a 3D point in its own camera frame (stored losslessly as 16-bit video).' },
], [
  ['t^{c}_{j}', 'PC time of frame j of camera c', 'ns'],
  ['Δ_c', 'clock offset of camera c', '73–116 ms'],
  ['**u**^{c}_{k}', 'ArUco corner k seen by camera c', 'px'],
  ['d_{epi}', 'distance of a point to its epipolar line', 'px'],
  ['j^{*}(n)', 'frame of camera c paired with ZED frame n', '—'],
  ['D_c(**u**)', 'stored depth at pixel **u** = (u, v); 0 = none', 'uint16'],
  ['s_c', 'depth unit (ZED and RealSense)', '1 mm'],
  ['K_c', 'intrinsics of that depth image', 'px'],
], 'ZED depth is aligned with the rectified left image (intrinsics in the sidecar); RealSense depth is in the depth sensor frame (848 × 480).');


// 7. RGB-D export
s = pres.addSlide({ masterName: 'STEP', sectionTitle: SEC });
stepTitle(s, 2, 'RGB-D streams and timestamps');
picture(s, 'step02_depth_streams.jpg', 0.55, 1.35, 7.7, 4.35, 'ZED left RGB, ZED SDK depth and both RealSense depth streams (near = red).');
label(s, 'FORMATS', 8.55, 1.4, 4);
bullets(s, ['RGB: H.264 MP4 (CRF 17), native 1920×1080, unrectified ZED images (match the calibration)',
  'Depth: FFV1 MKV, 16-bit lossless; ZED in mm (rectified left), RealSense raw Z16 (848×480)',
  'Every frame: PC-clock time (Unix ns) in _frames.tsv; RealSense also ZED-aligned time',
  'JSON sidecar: intrinsics, extrinsics, units, exposure, codec'], 8.55, 1.8, 4.3, 4.8, 14);

SEC = '3 · Detect & track';
pres.addSection({ title: SEC });
mathSlide(3, 'Detect & track: markers, One Euro filter, WiLoR', [
  { eq: 'd1', maxH: 0.7, text: 'Marker pose: the 24 mm square whose four corners re-project best into every camera that sees it; with one view, the two mirror solutions are separated by continuity.' },
  { eq: 'd2', text: 'One Euro filter OE(x): an adaptive low-pass; α(f) is the smoothing factor for cutoff frequency f.' },
  { eq: 'd3', text: 'The cutoff f_c rises with the filtered speed v̂: smooth when still, responsive when moving. Applied to each hand box (centre, width, height).' },
  { eq: 'd4', text: 'WiLoR: per-view MANO pose and shape, plus 21 keypoints with confidence, from the box enlarged by 10 %.' },
], [
  ['R_m, **t**_m', 'marker pose in the ZED-left frame', '—, m'],
  ['**c**_j', 'marker corners (±12, ±12, 0)', 'mm'],
  ['x_t', 'raw signal at time t (box coordinate)', 'px'],
  ['Δt', 'time between frames', '≈ 33 ms'],
  ['f_{min}', 'minimum cutoff', '1.5 Hz'],
  ['β', 'speed gain', '8'],
  ['f_d', 'cutoff for the speed estimate', '1 Hz'],
  ['θ_c, **s**_c', 'MANO pose (16 rotations), shape (10)', '—'],
  ['κ_{cj}', 'confidence of keypoint j', '0–1'],
], 'The same One Euro form is reused later on rotations (stage 4), arm joints (stage 4) and the object pose (stage 4).');


// 8. Sync + ArUco
s = pres.addSlide({ masterName: 'STEP', sectionTitle: SEC });
stepTitle(s, 3, 'ArUco marker poses and clock alignment');
picture(s, 'step03_aruco_markers.jpg', 0.55, 1.3, 6.7, 5.0);
label(s, 'CLOCK ALIGNMENT', 7.6, 1.4, 5);
bullets(s, ['RealSense frames lag the ZED by 73–116 ms in saved PC time; offset found by matching marker corners',
  'Fitted on alternating 2 s blocks, checked on held-out blocks; unvalidated cameras are excluded',
  'Mug runs: ZED only · gesture runs: d435n2 excluded'], 7.6, 1.8, 5.2, 2.5, 14);
label(s, 'MARKER POSES', 7.6, 4.3, 5);
bullets(s, ['24 mm DICT_4X4_50 markers, decoded IDs reviewed', 'Multi-view fit or tracked single view; nothing interpolated',
  'Object markers 2 + 1 (mug: 0 + 1); forearm markers 4, 5'], 7.6, 4.7, 5.2, 2.0, 14);

// 9. Hand detection
s = pres.addSlide({ masterName: 'STEP', sectionTitle: SEC });
stepTitle(s, 3, 'Hand detection and per-view MANO prediction');
picture(s, 'step04_hand_detection.jpg', 0.55, 1.35, 7.7, 4.35, 'Detections and WiLoR predictions in the four views.');
bullets(s, ['YOLO hand detector; boxes tracked and One Euro smoothed (1.5 Hz, β 8)', 'Box enlarged 10 % before WiLoR',
  'WiLoR → MANO shape, wrist rotation, 15 joint rotations per view', 'Confidence-scored 2D keypoints are the multi-view measurements'], 8.55, 1.5, 4.3, 4.8, 15);

SEC = '4 · Fit hand, arm & object';
pres.addSection({ title: SEC });
mathSlide(4, 'Fit (a): MANO hand model and multi-view cost', [
  { eq: 'm1', text: 'MANO: a hand mesh driven by 16 rotations (wrist + 15 finger joints) and 10 shape coefficients.', gap: 0.12 },
  { eq: 'm2', text: '3D joints at time t: the model joints placed so that the wrist sits at τ_t.', gap: 0.12 },
  { eq: 'm3', text: 'Cost minimised over θ_t and τ_t for all frames together (300 Adam iterations); **s** fixed per recording. L_{prior} = Σ‖R_{tk} − R^{WiLoR}_{tk}‖^2 keeps finger rotations near WiLoR.', gap: 0.12 },
  { eq: 'm4', maxH: 0.55, text: 'Data terms: visibility-weighted reprojection into every camera, and agreement with the triangulated joints.', gap: 0.12 },
  { eq: 'm5', maxH: 0.55, text: 'Temporal: each frame stays near the time-weighted interpolation of its neighbours. L_{finger} and L_{rot} do the same for wrist-relative joints (4 mm) and joint rotations (0.06).', gap: 0.12 },
], [
  ['M', 'MANO right-hand model', '—'],
  ['V, J', 'mesh vertices, joints (OpenPose order)', '778, 21'],
  ['R_0, R_k', 'wrist rotation, finger-joint rotations', 'SO(3)'],
  ['**s**', 'shape coefficients, one per recording', '10'],
  ['τ_t', 'wrist position', 'm'],
  ['**u**_{tcj}', '2D keypoint of joint j in camera c', 'px'],
  ['w_{tcj}', 'visibility weight: confidence^4 · finger chain · size', '0–1'],
  ['X̂_{tj}, r_{tj}', 'triangulated joint and its reliability', 'm, 0–1'],
  ['ρ_s', 'Huber loss at scale s', '—'],
  ['η_t', 'interpolation weight from the timestamps', '0–1'],
], 'Weights: L = L_2D + 0.35 L_3D + 0.15 L_prior + 4 L_world + 3 L_finger + 0.25 L_rot. Huber scales: 15 px (2D), 2 cm (3D), 6 mm (world), 4 mm (finger), 0.06 (rotation).');


// 10. MANO fit
s = pres.addSlide({ masterName: 'STEP', sectionTitle: SEC });
stepTitle(s, 4, 'Multi-view MANO fit and One Euro filtering');
picture(s, 'step05_mano_fit.jpg', 0.55, 1.35, 6.2, 3.5, 'Fitted MANO mesh on ZED left.');
picture(s, 'step05_filter_closeup.jpg', 6.95, 1.35, 5.9, 3.5, 'Close-up: before (orange) and after (green) filtering.');
bullets(s, ['Visibility-weighted triangulation, one shared shape per recording, temporal smoothness terms',
  'One Euro on SO(3): fingers 1 Hz/β 1, wrist 2.5 Hz/β 3; fingertip jitter 1.46 → 0.75 px',
  'Output: 21 joints + 778 vertices in the ZED-left frame; median reprojection ≈ 11 px'], 0.55, 5.35, 12.3, 1.5, 14);

mathSlide(4, 'Fit (b): rotation filtering and right arm', [
  { eq: 'f1', text: 'Each joint rotation\'s angular speed sets the One Euro cutoff (fingers 1 Hz, β 1; wrist 2.5 Hz, β 3).' },
  { eq: 'f2', text: 'Filtered rotation: move a fraction α along the shortest rotation from the previous estimate to the new one; the result is always a valid rotation.' },
  { eq: 'f3', text: 'Elbow from one view: the point on the camera ray at forearm length from the wrist (far intersection); the shoulder likewise from the elbow.' },
  { eq: 'f4', maxH: 0.65, text: '3D One Euro on elbow and shoulder, then the bone lengths are restored exactly.' },
  { eq: 'f5', text: 'Fallback when no camera sees the joints: shoulder at a prior, elbow by two-bone inverse kinematics bending down and to the right.' },
], [
  ['R_t, R̂_t', 'measured, filtered joint rotation', 'SO(3)'],
  ['ω_t', 'angular velocity', 'rad/s'],
  ['**o**_c, **d**_c', 'camera centre; unit ray through the keypoint', 'm; —'],
  ['b', '**d**_c · (**o**_c − **w**)', 'm'],
  ['**w**, **e**, **h**', 'wrist (MANO), elbow, shoulder', 'm'],
  ['L_f, L_u', 'forearm, upper-arm length', '0.27, 0.30 m'],
  ['OE', 'elbow 0.7 Hz / β 4; shoulder 0.3 Hz / β 2', '—'],
  ['**n̂**, **p̂**', 'shoulder→wrist direction; bend direction ⟂ n̂', '—'],
], 'Only d435n2 sees the shoulder, so most shoulder/elbow positions come from one view at fixed bone lengths; filters reset after gaps over 100 ms.');


// 11. Arm
s = pres.addSlide({ masterName: 'STEP', sectionTitle: SEC });
stepTitle(s, 4, 'Right arm: shoulder and elbow');
picture(s, 'step06_right_arm.jpg', 0.55, 1.35, 7.7, 4.35, 'Raw (orange) and filtered (green) arm in the four views.');
bullets(s, ['YOLO-pose keypoints, anchored on the MANO wrist', 'Only d435n2 sees the shoulder: single-view rays at 30 / 27 cm bone lengths',
  'Unseen joints: shoulder prior + two-bone IK (method code 4)', 'One Euro in 3D: jitter 7.2 → 0.7 mm (shoulder), 5.4 → 1.0 mm (elbow)'], 8.55, 1.5, 4.3, 4.8, 15);

mathSlide(4, 'Fit (c): object registration and pose', [
  { eq: 'o1', maxH: 0.65, text: 'Mesh-to-marker transform T*: the placement (and optional scale σ) whose projected outline best overlaps the object\'s colour mask in N sampled views.' },
  { eq: 'o2', text: 'Projected outline: convex hull of the mesh vertices projected into camera c at frame n.' },
  { eq: 'o3', maxH: 0.65, text: 'Rest prior: in the first frame the object stands upright (tilt φ) with its base on the table (lowest point h_{min}; coasters allowed); λ = 0.3.' },
  { eq: 'o4', text: 'Object pose over time: the filtered marker pose composed with T* (One Euro 1.5 Hz; β 20 per m/s, 8 per rad/s).' },
], [
  ['T*, σ', 'mesh-to-marker transform, mesh scale', '—'],
  ['V', 'STL mesh vertices', 'm'],
  ['S_{nc}', 'projected mesh outline', 'pixels'],
  ['M_{nc}', 'object colour mask (HSV range)', 'pixels'],
  ['N', 'views: 12 frames × cameras', '—'],
  ['T_{z←m}(t)', 'marker pose at time t', '—'],
  ['φ', 'tilt of mesh +Z from gravity', 'deg'],
  ['h_{min}', 'lowest mesh point above the table', 'cm'],
  ['IoU', 'intersection over union', '0–1'],
], 'Search: ~4,000 start placements (6 faces × 4 up-directions × positions × rolls) scored on a subset of views, the best 4 refined with Nelder–Mead.');


// 12. Object registration
s = pres.addSlide({ masterName: 'STEP', sectionTitle: SEC });
stepTitle(s, 4, 'Object registration: STL → marker');
picture(s, 'step07_mesh_fit.jpg', 0.55, 1.35, 12.2, 3.45, 'Mesh outline (magenta) projected through the marker pose in sampled frames.');
bullets(s, ['Colour mask of the object in 12 multi-view frames; projected mesh outline compared by IoU',
  '~4,000 start placements → Nelder–Mead on 6 pose (+ optional scale) parameters', 'Rest prior: object upright on the table at the start (YCB meshes: +Z up)'], 0.55, 5.1, 6.3, 1.8, 14);
card(s, 7.05, 5.1, 5.75, 1.6);
s.addText([{ text: 'Placement rule (mocap file)', options: { bold: true, color: HEX.dk2, breakLine: true } },
  { text: 'world_cm(v) = LOC + ROT·100·(SX vx, −SY vy, SZ vz)', options: { fontFace: 'Courier New', fontSize: 12 } }],
  { x: 7.2, y: 5.2, w: 5.5, h: 1.4, fontSize: 14, color: C.text1, valign: 'middle', isTextBox: true });

SEC = '5 · Export & visualize';
pres.addSection({ title: SEC });
mathSlide(5, 'Export & visualize: world frame and mocap file', [
  { eq: 'w1', text: 'World frame: A turns ZED-left axes into forward / right / up (Unreal, left-handed, det A = −1); origin **o** on the table below the ZED; centimetres.' },
  { eq: 'w2', maxH: 0.6, text: 'Bone axes: x points from the child joint back along the bone, z is the hint **k** made perpendicular, y completes the frame; q is its quaternion.' },
  { eq: 'w3', text: 'Object line: ROT and LOC place the STL in the world; F flips the right-handed mesh into the left-handed world.' },
  { eq: 'w4', text: 'Check: the exported poses re-projected into each camera (overlay video).' },
], [
  ['**f**, **g**', 'forward (towards d435n2), up (gravity, ZED IMU)', '—'],
  ['A', 'ZED-left → world rotation', 'det −1'],
  ['**o**, **n**_0, d_0', 'world origin; table normal and distance', 'm'],
  ['**p**_{child}', 'next joint along the chain', 'cm'],
  ['**k**', 'hint: elbow hinge (arm), index→little knuckle (hand, fingers), palm normal (thumb)', '—'],
  ['q', 'bone quaternion (x, y, z, w)', '—'],
  ['LOC, ROT', 'object location and quaternion in the file', 'cm; —'],
  ['σ', 'mesh scale (SCALE_X/Y/Z)', '—'],
  ['**û**_c', 're-projected pixel used by the overlay', 'px'],
], 'Mocap rows are written only for frames with a valid hand fit; FRAME_NUMBER is the ZED frame index.');

// 13. World + mocap
s = pres.addSlide({ masterName: 'STEP', sectionTitle: SEC });
stepTitle(s, 5, 'World frame and motion-capture export');
card(s, 0.55, 1.4, 7.1, 4.0, '1F2A30');
s.addText('{\n  version:4,\n  camera:{loc:[-0.94,0.40,47.64],rot:[...],fov:63.2},\n  objects:[ {mesh:mustard,loc:[...],rot:[...]} ]\n}\n# Format: TIMESTAMP|FRAME_NUMBER|FRAME_TIMESTAMP|HEAD...\n52.98|1530|52.98|-0.94|0.40|47.64|...|21|\n  clavicle_r|..|upperarm_r|..|lowerarm_r|..|hand_r|..\n  wrist_inner_r|..|thumb_01_r|.. ... |pinky_03_r|..\n  |1|object_0|loc|quat|SX|SY|SZ',
  { x: 0.75, y: 1.6, w: 6.75, h: 3.7, fontFace: 'Courier New', fontSize: 13, color: 'D8E6E8', valign: 'top', isTextBox: true });
s.addText('Excerpt (abbreviated) of sub-01_ses-01_task-mustardbottle_run-01_space-world_mocap.txt', { x: 0.55, y: 5.5, w: 7.1, h: 0.35, fontSize: 11, italic: true, color: '5A6870', margin: 0, isTextBox: true });
bullets(s, ['Unreal axes, cm: X forward (towards d435n2), Y right, Z up (ZED IMU gravity)', 'Origin: table plane below the ZED camera',
  '21 bones: clavicle → upper arm → forearm → hand, wrist inner/outer, 15 finger bones', 'Bone axes as in the reference file: −X to child, Z = hinge / lateral',
  'Object at the STL origin with fitted scale; One Euro on its pose'], 7.95, 1.5, 4.9, 5.0, 14);

// 14. Visualization
s = pres.addSlide({ masterName: 'STEP', sectionTitle: SEC });
stepTitle(s, 5, 'Visualization and quality control');
picture(s, 'anim_3d_viewer.gif', 0.55, 1.35, 6.0, 4.6, '3D viewer (HTML): MANO hand, arm and object mesh from the mocap file.');
picture(s, 'anim_overlay_four_views.gif', 6.8, 1.35, 6.0, 4.4, 'Overlay video: exported poses projected into all views.');
s.addText('Both read back the exported files, so a correct overlay also checks the export.', { x: 0.55, y: 6.35, w: 12.2, h: 0.4, fontSize: 13, italic: true, color: '5A6870', isTextBox: true });

// 15. Results table
pres.addSection({ title: 'Dataset' });
s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Dataset' });
s.addText('Recordings', { placeholder: 'title' });
const rec = [['Recording', 'Object', 's', 'Frames', 'Hand pose', 'Object pose', 'Views (validated timing)'],
  ['task-mug_run-01', 'mug', '83.5', '2,386', '1,360', '2,386', 'ZED L, R'], ['task-mug_run-02', 'mug', '104.0', '3,015', '1,956', '3,012', 'ZED L, R'],
  ['task-mustardbottle_run-01', 'mustard bottle', '107.5', '3,109', '2,686', '3,103', 'all 4'], ['task-banana_run-01', 'banana-shaped (no mesh)', '79.8', '2,311', '1,802', '2,309', 'all 4'],
  ['task-bowl_run-01', 'bowl', '92.5', '2,585', '1,476', '2,543', 'all 4'], ['task-puddingbox_run-01', 'pudding box', '102.1', '2,954', '2,199', '2,954', 'all 4'],
  ['task-crackerbox_run-01', 'cracker box (×1.2–1.28)', '88.3', '2,531', '1,695', '2,531', 'all 4'], ['task-gesture_run-01', '—', '36.6', '1,061', '732', '—', 'ZED L, R, d435n1'],
  ['task-gesture_run-02', '—', '101.4', '2,935', '2,357', '—', 'ZED L, R, d435n1']];
s.addTable(rec.map((r, i) => r.map((c, j) => ({ text: c, options: { bold: i === 0, align: j >= 2 && j <= 5 ? 'right' : 'left', color: i === 0 ? HEX.lt1 : HEX.dk1, fill: { color: i === 0 ? HEX.accent1 : (i % 2 ? 'FFFFFF' : HEX.lt2) } } }))),
  { x: 0.55, y: 1.35, w: 12.2, colW: [2.9, 2.4, 0.8, 1.05, 1.15, 1.25, 2.65], fontSize: 13, rowH: 0.42, border: { type: 'solid', pt: 0.5, color: 'D5DDE0' }, valign: 'middle' });
s.addText('All recordings: sub-01, ses-01 (2026-09-23). Object poses use the YCB mesh origin except the banana-shaped object (marker midpoint).', { x: 0.55, y: 5.75, w: 12.2, h: 0.6, fontSize: 13, italic: true, color: '5A6870', isTextBox: true });

// 16. Structure
s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Dataset' });
s.addText('Folder structure and file naming', { placeholder: 'title' });
card(s, 0.55, 1.35, 6.4, 4.6, '1F2A30');
s.addText('MCHOI_v1.0/\n├─ README.md, dataset_description.json\n├─ participants / tasks / cameras / recordings.tsv\n├─ calibration/  calib-01_rig_calibration.json\n│   └─ captures/ ..._pose-NNNN_cam-<cam>_charuco.png\n├─ objects/      obj-<object>_mesh.stl\n├─ sub-01/ses-01/video/\n│   ├─ <rec>_cam-<cam>_rgb.mp4 (+ .json, _frames.tsv)\n│   └─ <rec>_cam-<cam>_depth.mkv\n├─ derivatives/mchoi-pipeline/sub-01/ses-01/\n│   ├─ hand/ arm/ object/ mocap/ sync/ viz/\n└─ docs/   pipeline deck, figures, GIFs',
  { x: 0.7, y: 1.5, w: 6.15, h: 4.35, fontFace: 'Courier New', fontSize: 13, color: 'D8E6E8', valign: 'top', isTextBox: true });
label(s, 'NAMING (BIDS-STYLE ENTITIES)', 7.25, 1.4, 5.5);
s.addText([{ text: 'sub-01_ses-01_task-mug_run-02_cam-d435n1_depth.mkv', options: { fontFace: 'Courier New', bold: true, color: HEX.accent3, breakLine: true } },
  { text: ' ', options: { breakLine: true } },
  { text: 'sub', options: { bold: true } }, { text: ' participant · ', options: {} }, { text: 'ses', options: { bold: true } }, { text: ' session', options: { breakLine: true } },
  { text: 'task', options: { bold: true } }, { text: ' object or gesture · ', options: {} }, { text: 'run', options: { bold: true } }, { text: ' repetition', options: { breakLine: true } },
  { text: 'cam', options: { bold: true } }, { text: ' camera · ', options: {} }, { text: 'space', options: { bold: true } }, { text: ' zedleft (m) or world (cm)', options: { breakLine: true } },
  { text: 'desc', options: { bold: true } }, { text: ' derivative · suffix = data type (rgb, depth, handpose, mocap…)', options: {} }],
  { x: 7.25, y: 1.8, w: 5.6, h: 2.6, fontSize: 14, color: C.text1, valign: 'top', isTextBox: true });
label(s, 'OPEN FORMATS', 7.25, 4.55, 5.5);
bullets(s, ['TSV + JSON sidecars for tables and metadata', 'NPZ for MANO arrays; STL meshes; MP4 / lossless MKV video', 'Self-contained HTML viewer'], 7.25, 4.95, 5.6, 1.8, 14);

// 17. Limitations
s = pres.addSlide({ masterName: 'DARK_CONTENT', sectionTitle: 'Dataset' });
s.addText('Limitations and release checklist', { placeholder: 'title' });
s.addText([
  { text: 'Synchronization is content-based, not hardware-triggered; RealSense timing unvalidated for the mug runs', options: { bullet: true, breakLine: true } },
  { text: 'Shoulder / elbow are estimates (single view at fixed bone length, or prior + IK)', options: { bullet: true, breakLine: true } },
  { text: 'No hand–object contact constraint; scores are consistency measures, not ground truth', options: { bullet: true, breakLine: true } },
  { text: 'Banana-shaped object has no mesh; symmetric objects can face either way', options: { bullet: true } }],
  { x: 0.6, y: 1.7, w: 7.4, h: 4.2, fontSize: 17, color: 'E3EEF0', valign: 'top', paraSpaceAfter: 10, isTextBox: true });
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 8.4, y: 1.7, w: 4.35, h: 4.2, rectRadius: 0.08, fill: { color: '22434E' }, line: { color: '22434E' } });
s.addText([{ text: 'Before release', options: { bold: true, fontSize: 19, color: HEX.accent2, breakLine: true } },
  { text: 'Authors and citation', options: { bullet: true, breakLine: true } }, { text: 'Licence (data; YCB and MANO terms)', options: { bullet: true, breakLine: true } },
  { text: 'Ethics approval and participant consent', options: { bullet: true, breakLine: true } }, { text: 'Repository DOI', options: { bullet: true } }],
  { x: 8.6, y: 1.85, w: 4.0, h: 3.9, fontSize: 16, color: 'E3EEF0', valign: 'top', paraSpaceAfter: 8, isTextBox: true });

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log('wrote', OUT);
})();
