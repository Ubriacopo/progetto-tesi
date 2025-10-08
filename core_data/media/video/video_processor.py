import cv2
import mediapipe as mp
import numpy as np
from moviepy import VideoFileClip, ImageSequenceClip


def enforce_minimum_frames(frames, target=32):
    if len(frames) >= target:
        return frames
    return [frames[i] for i in np.linspace(0, len(frames) - 1, target, dtype=int)]


class VideoResampler:
    """
    Detect once (MediaPipe) -> track (OpenCV) -> compute RGB/BB change -> auto-threshold ->
    keep frames whose change exceeds threshold -> return a new MoviePy clip.

    Adapted from VATE for VATE
    """

    def __init__(self, detect_conf: float = 0.5, reduce_bbox: float = 0.10, min_frames: int = 32):
        self.detect_conf: float = detect_conf
        self.reduce_bbox: float = reduce_bbox
        # Face detection
        self.mp_fd = mp.solutions.face_detection.FaceDetection(self.detect_conf)
        self.min_frames = min_frames
        # Tracker (CSRT preferred; fall back to KCF)
        self.tracker_ctor = None
        for ctor in (
                "TrackerCSRT_create", "TrackerKCF_create", "legacy.TrackerCSRT_create", "legacy.TrackerKCF_create"
        ):
            self.tracker_ctor = getattr(cv2, ctor, None) if self.tracker_ctor is None else self.tracker_ctor

    @staticmethod
    def _calc_threshold(vec):
        v = np.asarray(vec, dtype=np.float32)
        v = v[np.isfinite(v)]

        if v.size == 0:
            return np.inf

        v = np.trim_zeros(v, trim='fb')

        if v.size == 0:
            return np.inf

        return (np.max(v) + np.mean(v)) / 4.0

    @staticmethod
    def _shrink_box(x, y, w, h, rf, W, H):
        nw = int(w * (1 - rf))
        nh = int(h * (1 - rf))
        nx = x + (w - nw) // 2
        ny = y + (h - nh) // 2
        nx, ny = max(0, nx), max(0, ny)
        return nx, ny, min(W, nx + nw), min(H, ny + nh)

    def _first_face_bbox(self, frame_rgb):
        res = self.mp_fd.process(frame_rgb)
        if not res.detections:
            return None
        rbb = res.detections[0].location_data.relative_bounding_box
        H, W = frame_rgb.shape[:2]
        x = int(rbb.xmin * W)
        y = int(rbb.ymin * H)
        w = int(rbb.width * W)
        h = int(rbb.height * H)
        x1, y1, x2, y2 = self._shrink_box(x, y, w, h, self.reduce_bbox, W, H)
        return x1, y1, x2 - x1, y2 - y1

    def _init_tracker(self, frame_bgr, bbox_xywh):
        if self.tracker_ctor is None:
            return None
        tracker = self.tracker_ctor()
        ok = tracker.init(frame_bgr, tuple(bbox_xywh))
        return tracker if ok else None

    @staticmethod
    def _roi_hist_rgb(roi_bgr):
        roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        h0 = cv2.normalize(cv2.calcHist([roi], [0], None, [256], [0, 256]), None).flatten()
        h1 = cv2.normalize(cv2.calcHist([roi], [1], None, [256], [0, 256]), None).flatten()
        h2 = cv2.normalize(cv2.calcHist([roi], [2], None, [256], [0, 256]), None).flatten()
        return h0, h1, h2

    def resample_clip(self, clip: VideoFileClip, keep_when_no_face=True, output_fps=None):
        """
        Returns an ImageSequenceClip keeping frames where (RGB-change > thrRGB) OR (BB-change > thrBB).
        - keep_when_no_face: if True, keeps frames when tracking/detection fail (so output isn’t empty).
        - output_fps: if None, uses input fps.
        """
        fps = clip.fps if hasattr(clip, "fps") and clip.fps else 25
        output_fps = output_fps or fps

        # Pull frames as numpy arrays (RGB) using get_frame at original fps
        n_frames = int(np.floor(clip.duration * fps))
        frames_rgb = [clip.get_frame(i / fps) for i in range(n_frames)]
        if not frames_rgb:
            return ImageSequenceClip([], fps=output_fps)

        # Detect once
        first_rgb = frames_rgb[0].astype(np.uint8)
        first_bgr = cv2.cvtColor(first_rgb, cv2.COLOR_RGB2BGR)
        bbox = self._first_face_bbox(first_rgb)

        tracker = self._init_tracker(first_bgr, bbox) if bbox is not None else None
        prev_bbox = None
        prev_hist = None

        vecRGB, vecBB = [], []
        kept_indices = []

        for idx, fr_rgb in enumerate(frames_rgb):
            fr_bgr = cv2.cvtColor(fr_rgb, cv2.COLOR_RGB2BGR)
            H, W = fr_bgr.shape[:2]

            # update bbox
            if tracker is not None:
                ok, trk = tracker.update(fr_bgr)
                if ok:
                    x, y, w, h = map(int, trk)
                    x1, y1, x2, y2 = self._shrink_box(x, y, w, h, self.reduce_bbox, W, H)
                    bbox_xyxy = (x1, y1, x2, y2)
                else:
                    bbox_xyxy = None
            else:
                # try sparse re-detect every ~15 frames
                bbox_xyxy = None
                if idx % 15 == 0:
                    bb0 = self._first_face_bbox(fr_rgb)
                    if bb0 is not None:
                        tracker = self._init_tracker(fr_bgr, bb0)
                        if tracker is not None:
                            # will take effect next iteration
                            pass

            # compute signals
            if bbox_xyxy is not None:
                x1, y1, x2, y2 = bbox_xyxy
                roi = fr_bgr[y1:y2, x1:x2]
                if roi.size == 0:
                    rgb_change = np.inf
                    bb_change = np.inf
                else:
                    h0, h1, h2 = self._roi_hist_rgb(roi)
                    if prev_hist is not None:
                        rgb_change = (np.linalg.norm(h0 - prev_hist[0])
                                      + np.linalg.norm(h1 - prev_hist[1])
                                      + np.linalg.norm(h2 - prev_hist[2])) / 3.0
                    else:
                        rgb_change = np.inf

                    if prev_bbox is not None:
                        bb_change = np.linalg.norm(np.array([x1, y1, x2, y2]) - np.array(prev_bbox, dtype=np.int32))
                    else:
                        bb_change = np.inf

                    prev_hist = (h0, h1, h2)
                    prev_bbox = (x1, y1, x2, y2)
            else:
                # no bbox this frame
                rgb_change = np.inf
                bb_change = np.inf
                if keep_when_no_face and prev_hist is None:
                    # keep initial frames to avoid empty output
                    kept_indices.append(idx)

            vecRGB.append(rgb_change)
            vecBB.append(bb_change)

        # thresholds
        thrRGB = self._calc_threshold(vecRGB)
        thrBB = self._calc_threshold(vecBB)

        # select frames
        for i, (r, b) in enumerate(zip(vecRGB, vecBB)):
            if np.isfinite(r) and r > thrRGB or np.isfinite(b) and b > thrBB:
                kept_indices.append(i)

        # always keep first and last for context
        if frames_rgb:
            kept_indices.extend([0, len(frames_rgb) - 1])

        kept_indices = sorted(set(kept_indices))

        # build output clip
        kept_frames = [frames_rgb[i] for i in kept_indices]

        # Enforce minimum
        kept_frames = enforce_minimum_frames(kept_frames, target=self.min_frames)
        # return ImageSequenceClip(kept_frames, fps=output_fps)
        return kept_frames
