import dataclasses
from abc import abstractmethod, ABC
from typing import Optional

import numpy as np

from common.data.eeg import EEG
from common.data.eeg.saliency_extractor import EEGFeatureExtractor


class Segmenter(ABC):
    """
    Segments input samples into timed ranges defined by predefined logic.
    """

    def __init__(self, max_length: int):
        self.max_length: int = max_length

    @abstractmethod
    def compute_segments(self, sample) -> list[tuple[int, int]]:
        pass


class FixedIntervalsSegmenter(Segmenter):
    def compute_segments(self, sample: EEG) -> list[tuple[int, int]]:
        # Good indicator of duration.
        length = sample.data.duration  # Given the length of the sample in time
        starts = np.arange(0, length, self.max_length).astype(int)
        stops = np.minimum(starts + self.max_length, length).astype(int)

        # Avoid overlaps (0s durations make no sense)
        overlapping = starts - stops == 0
        starts = starts[~overlapping]
        stops = stops[~overlapping]

        return list(zip(starts, stops))


@dataclasses.dataclass
class Segment:
    start: float | int
    stop: float | int
    category_code: str
    duration: float


@dataclasses.dataclass
class Anchor:
    start: float | int  # Starting time
    stop: float | int  # End time
    idx: int  # Its index in the segments


# TODO: Rename
class FeatureAndRandomLogUniformIntervalsSegmenter(Segmenter):
    SHORT_LENGTH = 4
    MEDIUM_LENGTH = 12
    LONG_LENGTH = 32

    SHORT_KEY = "S"
    MEDIUM_KEY = "M"
    LONG_KEY = "L"

    def __init__(self, min_length: int, max_length: int, num_segments: int,
                 jitter_frac: dict = None,
                 coverage_resolution_sec: float = 0.1, coverage_cap_k: int = 5,
                 same_iou_max=None, cross_iou_max: float = 0.7):
        super().__init__(max_length)
        # todo dataclass
        self.same_iou_max: dict = same_iou_max
        if self.same_iou_max is None:
            self.same_iou_max = {'S': 0.0, 'M': 0.35, 'L': 0.25}
        self.cross_iou_max: float = cross_iou_max

        self.jitter_frac: dict = jitter_frac
        if self.jitter_frac is None:
            self.jitter_frac = {'S': 0.0, 'M': 0.3, 'L': 0.4}

        self.coverage_resolution_sec = coverage_resolution_sec
        self.coverage_cap_K = coverage_cap_k

        self.num_segments: int = num_segments
        self.min_length: int = min_length

        self.max_attempts = 4  # After 4 you fail for a duration.

    def sample_duration_log_uniform(self):
        uniform = np.random.uniform(0., 1.)
        # Rescale in the MIN-MAX
        return float(
            np.exp(np.log(self.min_length) + uniform * (np.log(self.max_length) - np.log(self.min_length)))
        )

    def classify_duration(self, duration: float) -> str:
        return self.SHORT_KEY if duration < self.SHORT_LENGTH else self.MEDIUM_KEY if duration < self.MEDIUM_LENGTH else self.LONG_KEY

    def compute_segments(self, sample: EEG) -> list[tuple[float, float]]:
        buckets: list[Segment] = []
        t = sample.data.duration

        extractor = EEGFeatureExtractor(sample.data)
        # To times
        candidate_anchors = extractor.pick_segments(self.SHORT_LENGTH, 0.5) / sample.data.duration
        anchors = {self.SHORT_KEY: [], self.MEDIUM_KEY: [], self.LONG_KEY: [], }

        # Coverage tracker in seconds (no index math)
        num_slots = int(np.ceil(sample.data.duration / self.coverage_resolution_sec))
        coverages = {
            self.SHORT_KEY: np.zeros(num_slots, dtype=np.int16),
            self.MEDIUM_KEY: np.zeros(num_slots, dtype=np.int16),
            self.LONG_KEY: np.zeros(num_slots, dtype=np.int16),
        }

        for duration in [self.sample_duration_log_uniform() for _ in range(self.num_segments)]:
            key = self.classify_duration(duration)
            ok: bool = self.extract(
                eeg=sample,
                t=t, d=duration,
                type_key=key,
                candidate_anchors=candidate_anchors,
                anchors=anchors[key],
                segments=buckets,
                extraction_jitter=.1,
                coverage=coverages[key]
            )

            if not ok:
                print(f"Something went wrong for interval {duration} and duration will be discarded")
        return [(bucket.start, bucket.stop) for bucket in buckets]

    def decide_start_anchor(self, eeg: EEG, t: float, d: float, candidate_anchors: np.ndarray,
                            segments: list[Segment]) -> float:
        # returns start
        base_on_feature = np.random.random() < .5
        if base_on_feature:
            # todo keep the extracted segments stored as they are computed once
            selected_candidate = np.random.choice(np.ones(len(candidate_anchors)))
            candidate_anchors = np.delete(candidate_anchors, selected_candidate)
            # TODO CEnter crop o ltro boh che nes o
            return selected_candidate + int(d * eeg.fs) / 2
        else:
            return np.random.uniform(0., max(1e-9, t - d))

    def decide_start_dependent(self, eeg: EEG, t: float, d: float, anchors: list[int],
                               segments: list[Segment], jitter_frac: float) -> tuple[float, Optional[int]]:
        # Short segments. Potential Anchors
        potential_anchors = [
            Anchor(segment.start, segment.stop, idx)
            for idx, segment in enumerate(segments)
            if segment.category_code == self.SHORT_KEY
        ]
        # At random, we might want to go for non anchored. We for sure cannot go anchored if potential anchors are all taken.
        center_on_anchor = np.random.random() < .5 and len(potential_anchors) > len(anchors)
        if center_on_anchor:
            free_potential_anchors = [short.idx for short in potential_anchors if short.idx not in anchors]
            anchor: int = np.random.choice(free_potential_anchors)

            anchor_segment = segments[anchor]
            center = (anchor_segment.start + anchor_segment.stop) / 2 + np.random.uniform(-jitter_frac, jitter_frac) * d
            start = np.clip(center - d / 2, 0., max(0., t - d))
            return start, anchor

        else:  # Randomly pick a point
            # Completely random point
            return np.random.uniform(0., max(1e-9, t - d)), None

    @staticmethod
    def _iou_1d(x_start, x_stop, y_start, y_stop) -> float:
        inter = max(0.0, min(x_stop, y_stop) - max(x_start, y_start))
        if inter <= 0: return 0.0
        union = (x_stop - x_start) + (y_stop - y_start) - inter
        return inter / union if union > 0 else 0.

    # Intersection over Union (Set Theory).
    def ok_iou(self, start, stop, bucket_type, segments) -> bool:
        for i_start, i_stop, i_bucket_type, _ in segments:
            iou = self._iou_1d(start, stop, i_start, i_stop)
            # Matching type has to respect same IoU overlap threshold
            if bucket_type == i_bucket_type:
                if iou > self.same_iou_max[bucket_type]:
                    return False
            # Respect the cross IoU overlap threshold
            elif iou > self.cross_iou_max:
                return False

        return True

    def add_coverage(self, start, stop, coverage: np.ndarray) -> None:
        start_idx = int(np.floor(start / self.coverage_resolution_sec))
        stop_idx = int(np.ceil(stop / self.coverage_resolution_sec))
        coverage[start_idx:stop_idx] += 1

    def check_coverage(self, start, stop, coverage: np.ndarray) -> bool:
        start_index = int(np.floor(start / self.coverage_resolution_sec))
        stop_index = int(np.ceil(stop / self.coverage_resolution_sec))
        return (coverage[start_index:stop_index] < self.coverage_cap_K).all()

    def extract(self,
                # t is media time, d is extraction duration
                eeg: EEG,
                t: float, d: float,
                type_key: str,
                candidate_anchors: np.ndarray,
                anchors: list[int],
                # (start, stop, type, duration)
                segments: list[Segment],
                extraction_jitter: float,
                coverage: np.ndarray,
                attempt: int = 0
                ):
        """

        :param candidate_anchors:
        :param eeg:
        :param type_key:
        :param t:
        :param d:
        :param anchors:
        :param segments:
        :param extraction_jitter:
        :param coverage:
        :param attempt:
        :return:
        """
        # At chance center on short + jitter or just sample randomly
        if attempt > self.max_attempts:
            return False  # Could not extract

        anchor_appended = False
        if type_key == self.SHORT_KEY:
            # Short extraction.
            start = self.decide_start_anchor(eeg, t, d, candidate_anchors, segments, )
        else:
            start, anchor = self.decide_start_dependent(eeg, t, d, anchors, segments, self.jitter_frac[type_key])
            if anchor is not None:
                anchors.append(anchor)
                # In order to rollback if next checks fail
                anchor_appended = True

        # Calculate end point
        stop = start + d
        if not self.ok_iou(start, stop, type_key, segments) or not self.check_coverage(start, stop, coverage):
            if anchor_appended: anchors.pop()
            return self.extract(
                eeg, t, d, type_key, candidate_anchors, anchors, segments, extraction_jitter, coverage, attempt + 1
            )

        # Tiny jitter to avoid identical cuts.
        if extraction_jitter > 0:
            jitter = (np.random.uniform(-1, 1) * extraction_jitter) * d
            start = float(np.clip(start + jitter, 0., max(0., t - d)))
            stop = start + d

        segments.append(Segment(start, stop, type_key, d))
        self.add_coverage(start, stop, coverage)
        return True


class RandomizedSizeIntervalsSegmenter(Segmenter):

    def __init__(self, max_length: int, num_segments: int):
        super().__init__(max_length)
        self.num_segments: int = num_segments

    def compute_segments(self, sample: EEG) -> list[tuple[float, float]]:
        segments = []
        # TODO Sbaglaito su start
        #   Ma con shutils funziona bene pare.
        for _ in range(self.num_segments):
            # Random duration: 0.5–30 s, expressed in samples.
            dur = np.random.randint(int(0.5 * sample.fs), int(self.max_length * sample.fs))
            # Pick a valid start (so stop doesn’t exceed max_length)
            max_start = int((sample.data.duration - self.max_length) * sample.fs) - dur
            start = np.random.randint(0, max_start)
            stop = start + dur
            # Convert to seconds
            segments.append((start / sample.fs, stop / sample.fs))
        return segments
