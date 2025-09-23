from __future__ import annotations
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
class Anchor:
    start: float | int  # Starting time
    stop: float | int  # End time
    idx: int  # Its index in the segments


@dataclasses.dataclass
class Feature:
    key: str
    max_length: int
    same_iou_max: float
    cross_iou_max: float
    jitter_frac: float

    def __lt__(self, other):
        return self.max_length < other.max_length


@dataclasses.dataclass
class Segment:
    start: float | int
    stop: float | int
    feature_spec: Feature
    duration: float


SHORT_FEATURE = Feature("S", 4, .0, .7, .0)
MEDIUM_FEATURE = Feature("M", 16, .35, .7, .3)
LONG_FEATURE = Feature("L", 32, .25, .7, .4)


# TODO: Rename
class FeatureAndRandomLogUniformIntervalsSegmenter(Segmenter):
    def __init__(self,
                 min_length: int,
                 max_length: int,
                 num_segments: int,
                 anchor_identification_hop: float,
                 *features: Feature,
                 coverage_resolution_sec: float = 0.1,
                 coverage_cap_k: int = 5,
                 return_seconds: bool = True,
                 extraction_jitter: float = .1,
                 ):
        """

        :param min_length:
        :param max_length:
        :param num_segments: How many segments we aim to generate. It might not always be possible for time constraints.
        (SHORT segments are non overlapping by definition in SHORT_FEATURE).
        :param anchor_identification_hop:
        :param jitter_frac:
        :param coverage_resolution_sec:
        :param coverage_cap_k:
        :param same_iou_max:
        :param cross_iou_max:
        :param return_seconds:
        :param extraction_jitter:
        """
        super().__init__(max_length)
        self.anchor_identification_hop: float = anchor_identification_hop
        default_features = [SHORT_FEATURE, MEDIUM_FEATURE, LONG_FEATURE]

        self.features_specs: list[Feature] = default_features
        if features is not None and len(features) > 0:
            self.features_specs = list(features)
        # Sort for increasing duration
        self.features_specs = sorted(self.features_specs)
        # Shortest is considered our anchoring method.
        self.anchor_modality = self.features_specs[0]

        # TODO a che servono?
        self.coverage_resolution_sec: float = coverage_resolution_sec
        self.coverage_cap_K: int = coverage_cap_k

        self.num_segments: int = num_segments
        self.min_length: int = min_length

        self.max_attempts = 10  # After 4 you fail for a duration.
        self.return_seconds: bool = return_seconds  # If False return sampled points.
        self.extraction_jitter: float = extraction_jitter

    def sample_duration_log_uniform(self):
        uniform = np.random.uniform(0., 1.)
        duration = np.exp(np.log(self.min_length) + uniform * (np.log(self.max_length) - np.log(self.min_length)))
        return float(duration)

    def classify_duration(self, duration: float) -> Feature:
        for feature in self.features_specs:
            if duration < feature.max_length:
                return feature
        raise ValueError("Extracted segment is not a valid duration for the given specs")

    def compute_segments(self, sample: EEG) -> list[tuple[float, float]]:
        buckets: list[Segment] = []
        t = sample.data.duration

        extractor = EEGFeatureExtractor(sample.data)
        candidate_anchors = extractor.pick_segments(
            self.anchor_modality.max_length, self.anchor_identification_hop,
            bands=((4, 8), (8, 13), (13, 30)), band_weights=(0.4, 0.5, 0.4)
        )
        # todo vedi di far lavorare solo con punti e non secondi qui
        num_slots = int(np.ceil(sample.data.duration / self.coverage_resolution_sec))
        anchors = {spec.key: [] for spec in self.features_specs}
        # Coverage tracker in seconds (no index math)
        coverages = {spec.key: np.zeros(num_slots, dtype=np.int16) for spec in self.features_specs}

        for duration in sorted([self.sample_duration_log_uniform() for _ in range(self.num_segments)]):
            feature = self.classify_duration(duration)
            ok, candidate_anchors = self.extract(
                eeg=sample,
                t=t, d=duration,
                base_feature=feature,
                candidate_anchors=candidate_anchors,
                anchors=anchors[feature.key],
                segments=buckets,
                coverage=coverages[feature.key]
            )

            if not ok:
                print(f"Something went wrong for interval {duration} and duration will be discarded")

        if self.return_seconds:
            return [(bucket.start / sample.fs, bucket.stop / sample.fs) for bucket in buckets]
        return [(bucket.start, bucket.stop) for bucket in buckets]

    def decide_start_anchor(self, eeg: EEG, t: float, d: float, candidate_anchors: np.ndarray) \
            -> tuple[float, float, Optional[int]]:
        # Returns start. We favour more based on features than on random selection here.
        base_on_feature = np.random.random() < .6 and len(candidate_anchors) > 0
        if base_on_feature:
            selected_candidate = np.random.choice(len(candidate_anchors))
            # For now keep it simple: All our segments have a duration of 4 seconds if short.
            start = int(candidate_anchors[selected_candidate])
            return start, start + self.anchor_modality.max_length * eeg.fs, selected_candidate
        else:
            start = int(np.random.uniform(0., max(1e-9, (t - d) * eeg.fs)))
            return start, int(start + d * eeg.fs), None

    def decide_start_dependent(self, eeg: EEG, t: float, d: float, anchors: list[int],
                               segments: list[Segment], jitter_frac: float) -> tuple[float, float, Optional[int]]:
        # Short segments. Potential Anchors
        potential_anchors = [
            Anchor(segment.start, segment.stop, idx)
            for idx, segment in enumerate(segments)
            if segment.feature_spec == self.anchor_modality.max_length
        ]

        # At random, we might want to go for non anchored.
        # We for sure cannot go anchored if potential anchors are all taken.
        center_on_anchor = np.random.random() < .5 and len(potential_anchors) > len(anchors)
        if center_on_anchor:
            free_potential_anchors = [short.idx for short in potential_anchors if short.idx not in anchors]
            anchor: int = np.random.choice(free_potential_anchors)

            anchor_segment = segments[anchor]
            center = (anchor_segment.start + anchor_segment.stop) / 2 + np.random.uniform(-jitter_frac, jitter_frac) * d
            start = int(np.clip(center - d / 2 * eeg.fs, 0, max(0, int((t - d) * eeg.fs))))
            return start, int(start + d * eeg.fs), anchor

        else:  # Randomly pick a point
            # Completely random point
            start = int(np.random.uniform(0, max(1e-9, (t - d) * eeg.fs)))
            return start, int(start + d * eeg.fs), None

    @staticmethod
    def _iou_1d(x_start, x_stop, y_start, y_stop) -> float:
        inter = max(0.0, min(x_stop, y_stop) - max(x_start, y_start))
        if inter <= 0: return 0.0
        union = (x_stop - x_start) + (y_stop - y_start) - inter
        return inter / union if union > 0 else 0.

    # Intersection over Union (Set Theory).
    def ok_iou(self, start: float, stop: float, bucket_type: Feature, segments: list[Segment], ) -> bool:
        for segment in segments:
            iou = self._iou_1d(start, stop, segment.start, segment.stop)
            # Matching type has to respect same IoU overlap threshold
            if bucket_type == segment.feature_spec:
                if iou > bucket_type.same_iou_max:
                    return False
            # Respect the cross IoU overlap threshold
            elif iou > bucket_type.cross_iou_max:
                return False

        return True

    def add_coverage(self, eeg: EEG, start, stop, coverage: np.ndarray) -> None:
        coverage_chunks = self.coverage_resolution_sec / eeg.fs
        start_idx = int(np.floor(start / coverage_chunks))
        stop_idx = int(np.ceil(stop / coverage_chunks))
        coverage[start_idx:stop_idx] += 1

    def check_coverage(self, eeg: EEG, start, stop, coverage: np.ndarray) -> bool:
        coverage_chunks = self.coverage_resolution_sec / eeg.fs
        start_index = int(np.floor(start / coverage_chunks))
        stop_index = int(np.ceil(stop / coverage_chunks))
        return (coverage[start_index:stop_index] < self.coverage_cap_K).all()

    def extract(self,
                # t is media time, d is extraction duration
                eeg: EEG,
                t: float, d: float,
                base_feature: Feature,
                candidate_anchors: np.ndarray,
                anchors: list[int],
                # (start, stop, type, duration)
                segments: list[Segment],
                coverage: np.ndarray,
                attempt: int = 0
                ):
        """

        :param candidate_anchors:
        :param eeg:
        :param base_feature:
        :param t:
        :param d:
        :param anchors:
        :param segments:
        :param coverage:
        :param attempt:
        :return:
        """
        # At chance center on short + jitter or just sample randomly
        if attempt > self.max_attempts:
            return False, candidate_anchors  # Could not extract

        extracted_anchor: Optional[int] = None
        reference_anchor: Optional[int] = None
        if base_feature == self.anchor_modality:
            # Short extraction.
            start, stop, extracted_anchor = self.decide_start_anchor(eeg, t, d, candidate_anchors)
        else:
            # Longer segments extraction.
            jitter = base_feature.jitter_frac
            start, stop, reference_anchor = self.decide_start_dependent(eeg, t, d, anchors, segments, jitter)

        # Tiny jitter to avoid identical cuts
        if self.extraction_jitter > 0:
            jitter = np.random.uniform(-1, 1) * self.extraction_jitter
            jitter *= d * eeg.fs  # To points of EEG
            # Get start and stop points.
            start = int(np.clip(start + jitter, 0, max(0, eeg.fs * int(t - d))))
            stop = start + d * eeg.fs

        ok_iou = self.ok_iou(start, stop, base_feature, segments)
        if not ok_iou or not self.check_coverage(eeg, start, stop, coverage):
            print(
                f"Check failed for ({start}-{stop}) ({base_feature.key}).\n"
                f"Problem was: {'IoU' if not ok_iou else 'coverage'}.\n"
                f"It generated from {'extraction' if extracted_anchor is not None else 'segment/random'}.\n\n")
            return self.extract(eeg, t, d, base_feature, candidate_anchors, anchors, segments, coverage, attempt + 1)

        segments.append(Segment(start=start, stop=stop, feature_spec=base_feature, duration=d))
        self.add_coverage(eeg, start, stop, coverage)

        if extracted_anchor is not None:
            # Remove extracted element as it was taken.
            print(f"We used anchor: {candidate_anchors[extracted_anchor]} ({extracted_anchor}).\n"
                  f"candidate_anchors: {candidate_anchors}.\n\n")
            candidate_anchors = np.delete(candidate_anchors, extracted_anchor)
        if reference_anchor is not None:
            # This anchor was used so we "register" its usage
            anchors.append(reference_anchor)

        return True, candidate_anchors


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
