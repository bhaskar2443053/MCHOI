# Changelog

## 1.0.0 (2026-10-06)

First public version of the code, matching the MCHOI v1.0 dataset release (2026-10-02).

- Package `mchoi` (src layout) replacing the earlier `tools/` scripts: ArUco, hand, arm, scene, export and dataset stages,
  plus the calibration and recording apps.
- `mchoi.pipeline` processes one recording; `mchoi.reproduce` rebuilds the whole release from `configs/mchoi_v1.0.json`.
- Locations come from `configs/paths.local.json` or `MCHOI_*` environment variables (`src/mchoi/paths.py`).
- Pinned environments in `environment/`, setup scripts and a verified model downloader in `scripts/`.
- Pipeline slides built from `deck/` without external tooling beyond Node.js and pptxgenjs.
