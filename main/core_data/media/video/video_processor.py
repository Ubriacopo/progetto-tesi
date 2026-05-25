import cv2
import mediapipe as mp
import numpy as np
from moviepy import ImageSequenceClip

from main.core_data.media.video.utils import VideoTensor


def enforce_minimum_frames(frames, target=32):
    if len(frames) >= target:
        return frames
    return [frames[i] for i in np.linspace(0, len(frames) - 1, target, dtype=int)]


class VideoResampler:
    """
    Detect once (MediaPipe Tasks API) -> track (OpenCV) -> compute RGB/BB change ->
    auto-threshold -> keep frames whose change exceeds threshold.
    """

    def __init__(self, detect_conf: float = 0.5, reduce_bbox: float = 0.10,
                 min_frames: int = 32, model_path: str = "/home/jfichera/progetto-tesi/models/face_detector.task", ):
        self.detect_conf = detect_conf
        self.reduce_bbox = reduce_bbox
        self.min_frames = min_frames

        # MediaPipe 0.10.35 Tasks API
        BaseOptions = mp.tasks.BaseOptions
        FaceDetector = mp.tasks.vision.FaceDetector
        FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
            min_detection_confidence=detect_conf,
        )

        self.mp_fd = FaceDetector.create_from_options(options)

        # Tracker (CSRT preferred; fall back to KCF)
        self.tracker_ctor = None
        for ctor_path in (
                ("TrackerCSRT_create",),
                ("TrackerKCF_create",),
                ("legacy", "TrackerCSRT_create"),
                ("legacy", "TrackerKCF_create"),
        ):
            obj = cv2
            for name in ctor_path:
                obj = getattr(obj, name, None)
                if obj is None:
                    break

            if obj is not None:
                self.tracker_ctor = obj
                break

    def close(self):
        if self.mp_fd is not None:
            self.mp_fd.close()
            self.mp_fd = None

    @staticmethod
    def _calc_threshold(vec):
        v = np.asarray(vec, dtype=np.float32)
        v = v[np.isfinite(v)]

        if v.size == 0:
            return np.inf

        v = np.trim_zeros(v, trim="fb")

        if v.size == 0:
            return np.inf

        return (np.max(v) + np.mean(v)) / 4.0

    @staticmethod
    def _shrink_box(x, y, w, h, rf, W, H):
        nw = int(w * (1 - rf))
        nh = int(h * (1 - rf))
        nx = x + (w - nw) // 2
        ny = y + (h - nh) // 2

        x1 = max(0, nx)
        y1 = max(0, ny)
        x2 = min(W, nx + nw)
        y2 = min(H, ny + nh)

        return x1, y1, x2, y2

    def _first_face_bbox(self, frame_rgb):
        frame_rgb = np.ascontiguousarray(frame_rgb.astype(np.uint8))

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb,
        )

        result = self.mp_fd.detect(mp_image)

        if not result.detections:
            return None

        bbox = result.detections[0].bounding_box

        H, W = frame_rgb.shape[:2]

        x = max(0, int(bbox.origin_x))
        y = max(0, int(bbox.origin_y))
        w = max(0, int(bbox.width))
        h = max(0, int(bbox.height))

        x1, y1, x2, y2 = self._shrink_box(x, y, w, h, self.reduce_bbox, W, H)

        if x2 <= x1 or y2 <= y1:
            return None

        return x1, y1, x2 - x1, y2 - y1

    def _init_tracker(self, frame_bgr, bbox_xywh):
        if self.tracker_ctor is None or bbox_xywh is None:
            return None

        tracker = self.tracker_ctor()
        ok = tracker.init(frame_bgr, tuple(map(int, bbox_xywh)))
        return tracker if ok else None

    @staticmethod
    def _roi_hist_rgb(roi_bgr):
        roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        h0 = cv2.normalize(cv2.calcHist([roi], [0], None, [256], [0, 256]), None).flatten()
        h1 = cv2.normalize(cv2.calcHist([roi], [1], None, [256], [0, 256]), None).flatten()
        h2 = cv2.normalize(cv2.calcHist([roi], [2], None, [256], [0, 256]), None).flatten()
        return h0, h1, h2

    def resample_clip(self, clip: VideoTensor, keep_when_no_face=True, output_fps=None):
        fps = clip.fps if hasattr(clip, "fps") and clip.fps else 25
        output_fps = output_fps or fps

        frames_rgb = clip.value.numpy().transpose(0, 2, 3, 1)

        if len(frames_rgb) == 0 or not frames_rgb.any():
            return ImageSequenceClip([], fps=output_fps)

        first_rgb = frames_rgb[0].astype(np.uint8)
        first_bgr = cv2.cvtColor(first_rgb, cv2.COLOR_RGB2BGR)

        bbox = self._first_face_bbox(first_rgb)
        tracker = self._init_tracker(first_bgr, bbox) if bbox is not None else None

        prev_bbox = None
        prev_hist = None

        vecRGB, vecBB = [], []
        kept_indices = []

        for idx, fr_rgb in enumerate(frames_rgb):
            fr_rgb = fr_rgb.astype(np.uint8)
            fr_bgr = cv2.cvtColor(fr_rgb, cv2.COLOR_RGB2BGR)

            H, W = fr_bgr.shape[:2]
            bbox_xyxy = None

            if tracker is not None:
                ok, trk = tracker.update(fr_bgr)
                if ok:
                    x, y, w, h = map(int, trk)
                    x1, y1, x2, y2 = self._shrink_box(x, y, w, h, self.reduce_bbox, W, H)
                    bbox_xyxy = (x1, y1, x2, y2)
                else:
                    tracker = None

            if tracker is None and idx % 15 == 0:
                bb0 = self._first_face_bbox(fr_rgb)
                if bb0 is not None:
                    tracker = self._init_tracker(fr_bgr, bb0)
                    x, y, w, h = bb0
                    bbox_xyxy = (x, y, x + w, y + h)

            if bbox_xyxy is not None:
                x1, y1, x2, y2 = bbox_xyxy
                roi = fr_bgr[y1:y2, x1:x2]

                if roi.size == 0:
                    rgb_change = np.inf
                    bb_change = np.inf
                else:
                    h0, h1, h2 = self._roi_hist_rgb(roi)

                    if prev_hist is not None:
                        rgb_change = (
                                             np.linalg.norm(h0 - prev_hist[0])
                                             + np.linalg.norm(h1 - prev_hist[1])
                                             + np.linalg.norm(h2 - prev_hist[2])
                                     ) / 3.0
                    else:
                        rgb_change = np.inf

                    if prev_bbox is not None:
                        bb_change = np.linalg.norm(
                            np.array([x1, y1, x2, y2], dtype=np.int32)
                            - np.array(prev_bbox, dtype=np.int32)
                        )
                    else:
                        bb_change = np.inf

                    prev_hist = (h0, h1, h2)
                    prev_bbox = (x1, y1, x2, y2)
            else:
                rgb_change = np.inf
                bb_change = np.inf

                if keep_when_no_face and prev_hist is None:
                    kept_indices.append(idx)

            vecRGB.append(rgb_change)
            vecBB.append(bb_change)

        thrRGB = self._calc_threshold(vecRGB)
        thrBB = self._calc_threshold(vecBB)

        for i, (r, b) in enumerate(zip(vecRGB, vecBB)):
            if (np.isfinite(r) and r > thrRGB) or (np.isfinite(b) and b > thrBB):
                kept_indices.append(i)

        kept_indices.extend([0, len(frames_rgb) - 1])
        kept_indices = sorted(set(kept_indices))

        kept_frames = frames_rgb[kept_indices]
        kept_frames = enforce_minimum_frames(kept_frames, target=self.min_frames)

        return kept_frames
