"""Adapter for the original WiLoR model and its hand-specific YOLO checkpoint."""
import os
from pathlib import Path
import sys

import numpy as np


class WilorBackend:
    def __init__(self, source_dir, weights_dir, mano_dir, device='cuda', fp16=True,
                 confidence=0.3, detector_size=640):
        import torch
        os.environ['YOLO_AUTOINSTALL'] = 'false'
        if os.name == 'nt':
            # Upstream defaults to EGL, which is not the Windows OpenGL backend.
            os.environ.setdefault('PYOPENGL_PLATFORM', 'win32')
        sys.path.insert(0, str(Path(source_dir).resolve()))
        from wilor.configs import get_config
        from wilor.models import WiLoR
        from wilor.datasets.vitdet_dataset import ViTDetDataset
        from ultralytics import YOLO
        if device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; install CUDA PyTorch or pass --device cpu')
        self.device = torch.device(device)
        self.fp16 = fp16 and self.device.type == 'cuda'
        self.dtype = torch.float16 if self.fp16 else torch.float32
        self.confidence, self.detector_size = confidence, detector_size
        self.dataset_type = ViTDetDataset
        source_dir, weights_dir, mano_dir = map(lambda p: Path(p).resolve(), (source_dir, weights_dir, mano_dir))
        if not (mano_dir / 'MANO_RIGHT.pkl').is_file():
            raise FileNotFoundError(mano_dir / 'MANO_RIGHT.pkl')
        cfg = get_config(str(source_dir / 'pretrained_models/model_config.yaml'))
        cfg.defrost()
        cfg.MODEL.BACKBONE.pop('PRETRAINED_WEIGHTS', None)
        cfg.MODEL.BBOX_SHAPE = [192, 256]
        cfg.MANO.DATA_DIR = str(mano_dir)
        cfg.MANO.MODEL_PATH = str(mano_dir)
        cfg.MANO.MEAN_PARAMS = str(source_dir / 'mano_data/mano_mean_params.npz')
        cfg.freeze()
        print('Loading original WiLoR and local MANO_RIGHT.pkl...', flush=True)
        self.model = WiLoR(cfg, init_renderer=False)
        # The public upstream checkpoint is a legacy Lightning pickle.
        checkpoint = torch.load(weights_dir / 'wilor_final.ckpt', map_location='cpu', weights_only=False)
        state = checkpoint['state_dict']
        # Keep the user's MANO model, rather than replacing its buffers from the checkpoint.
        state = {k: v for k, v in state.items() if not k.startswith('mano.')}
        incompatible = self.model.load_state_dict(state, strict=False)
        missing = [k for k in incompatible.missing_keys if not k.startswith('mano.')]
        if missing or incompatible.unexpected_keys:
            raise RuntimeError(f'Checkpoint mismatch: missing={missing}, extra={incompatible.unexpected_keys}')
        del checkpoint, state
        self.model = self.model.to(device=self.device, dtype=self.dtype).eval()
        self.cfg = cfg
        self.faces = self.model.mano.faces.astype(np.int32)
        self.detector = YOLO(str(weights_dir / 'detector.pt')).to(self.device)
        if sorted(self.detector.names) != [0, 1]:
            raise ValueError(f'Expected WiLoR left/right hand detector, got {self.detector.names}')
        print(f'YOLO classes: {self.detector.names}; device: {self.device}; fp16: {self.fp16}', flush=True)

    def detect(self, bgr):
        return self.detect_with_keypoints(bgr)[0]

    def detect_with_keypoints(self, bgr):
        result = self.detector.predict(bgr, conf=self.confidence, imgsz=self.detector_size,
                                       iou=0.5, agnostic_nms=True, verbose=False, device=str(self.device))[0]
        return (result.boxes.data.detach().cpu().numpy()[:, :6],
                result.keypoints.data.detach().cpu().numpy())

    def reconstruct(self, bgr, tracks):
        return self.reconstruct_many([bgr], [tracks])[0]

    def reconstruct_many(self, images, track_lists):
        """Batch crops across equal-resolution camera views in a single model pass."""
        import torch
        from torch.utils.data import default_collate
        counts = [len(t) for t in track_lists]
        tracks = [t for group in track_lists for t in group]
        if not tracks:
            return [[] for _ in images]
        bgr = images[0]
        if any(image.shape != bgr.shape for image in images):
            raise ValueError('Batched camera views must have equal resolution')
        items = []
        for image, group in zip(images, track_lists):
            if not group:
                continue
            boxes = np.asarray([t['expanded_box'] for t in group], dtype=np.float32)
            right = np.asarray([t['is_right'] for t in group], dtype=np.float32)
            # Boxes are already expanded; disable the upstream 2.5x default.
            dataset = self.dataset_type(self.cfg, image, boxes, right, rescale_factor=1.0, fp16=self.fp16)
            items.extend(dataset[i] for i in range(len(dataset)))
        batch = default_collate(items)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.inference_mode():
            output = self.model(batch)
        cams = output['pred_cam'].float().cpu().numpy()
        joints = output['pred_keypoints_3d'].float().cpu().numpy()
        vertices = output['pred_vertices'].float().cpu().numpy()
        params = {k: v.float().cpu().numpy() for k, v in output['pred_mano_params'].items()}
        centers = batch['box_center'].float().cpu().numpy()
        sizes = batch['box_size'].float().cpu().numpy()
        height, width = bgr.shape[:2]
        focal = float(self.cfg.EXTRA.FOCAL_LENGTH / self.cfg.MODEL.IMAGE_SIZE * max(width, height))
        results = []
        for i, track in enumerate(tracks):
            # WiLoR predicts a canonical right hand. Undo the input flip for left hands.
            mirror = 2 * track['is_right'] - 1
            cams[i, 1] *= mirror
            joints[i, :, 0] *= mirror
            vertices[i, :, 0] *= mirror
            bs = float(sizes[i] * cams[i, 0])
            if not np.isfinite(bs) or bs <= 0:
                raise ValueError('WiLoR predicted an invalid camera scale')
            translation = np.array([2 * (centers[i, 0] - width / 2) / bs + cams[i, 1],
                                    2 * (centers[i, 1] - height / 2) / bs + cams[i, 2], 2 * focal / bs], dtype=np.float32)
            camera_joints = joints[i] + translation
            camera_vertices = vertices[i] + translation
            joints_2d = focal * camera_joints[:, :2] / camera_joints[:, 2:] + [width / 2, height / 2]
            vertices_2d = focal * camera_vertices[:, :2] / camera_vertices[:, 2:] + [width / 2, height / 2]
            square = np.r_[centers[i] - sizes[i] / 2, centers[i] + sizes[i] / 2]
            result = dict(**track, joints_2d=joints_2d.astype(np.float32), joints_3d=joints[i],
                          vertices=vertices[i], vertices_2d=vertices_2d.astype(np.float32),
                          camera_translation=translation, focal_length=focal, input_square=square,
                          mano_global_orient=params['global_orient'][i],
                          mano_hand_pose=params['hand_pose'][i], mano_betas=params['betas'][i])
            for name, value in result.items():
                if isinstance(value, np.ndarray) and not np.isfinite(value).all():
                    raise ValueError(f'Nonfinite WiLoR output: {name}')
            results.append(result)
        offsets = np.r_[0, np.cumsum(counts)]
        return [results[a:b] for a, b in zip(offsets, offsets[1:])]
