import dataclasses
from abc import abstractmethod, ABC
from typing import Optional

import numpy as np

from common.data.eeg import EEG


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
                anchors=anchors[key],
                segments=buckets,
                extraction_jitter=.1,
                coverage=coverages[key]
            )

            if not ok:
                print(f"Something went wrong for interval {duration} and duration will be discarded")
        return [(bucket.start, bucket.stop) for bucket in buckets]

    def decide_start_anchor(self, eeg: EEG, t: float, d: float, segments: list[Segment]) -> float:
        # returns start
        base_on_feature = np.random.random() < .5
        if base_on_feature:
            pass  # TODO extract
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
                anchors: list[int],
                # (start, stop, type, duration)
                segments: list[Segment],
                extraction_jitter: float,
                coverage: np.ndarray,
                attempt: int = 0
                ):
        """

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
            start = self.decide_start_anchor(eeg, t, d, segments, )
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
            return self.extract(eeg, t, d, type_key, anchors, segments, extraction_jitter, coverage, attempt + 1)

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


# TODO Valuta quello che ce qua sotto
import numpy as np
from scipy.signal import stft


def preprocess_raw(raw, l_freq=0.5, h_freq=40, notch=50):
    raw = raw.copy().load_data()
    raw.filter(l_freq, h_freq, phase='zero-double', verbose=False)
    raw.notch_filter([notch, 2 * notch], phase='zero-double', verbose=False)
    raw.set_eeg_reference('average')
    raw.apply_function(lambda x: (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-9))
    return raw


def artifact_weights_from_mne(raw, short_s, hop_s):
    # hard: flat/rails; soft: muscle
    ann = []
    try:
        ann.append(mne.preprocessing.annotate_flat(raw))
    except Exception:
        pass
    try:
        a_muscle, _ = mne.preprocessing.annotate_muscle_zscore(raw, ch_type='eeg', threshold=4.0,
                                                               filter_freq=(30., 80.), min_length_good=0.2)
        ann.append(a_muscle)
    except Exception:
        pass
    if ann:
        A = ann[0]
        for a in ann[1:]: A += a
        raw.set_annotations(A)

    fs = raw.info['sfreq']
    win = int(round(short_s * fs));
    hop = int(round(hop_s * fs))
    nF = 1 + (raw.n_times - win) // hop if raw.n_times >= win else 0
    starts = np.arange(nF) * hop
    w = np.ones(nF)
    hard = np.zeros(nF, dtype=bool)
    for on, dur, desc in zip(raw.annotations.onset, raw.annotations.duration, raw.annotations.description):
        s = int(round((on - raw.first_time) * fs));
        e = s + int(round(dur * fs))
        i0 = max(0, (s - win) // hop + 1 if s > 0 else 0);
        i1 = min(nF, e // hop + 1)
        if 'BAD_flat' in desc or 'BAD' in desc and 'muscle' not in desc:
            hard[i0:i1] = True
        if 'muscle' in desc:
            w[i0:i1] = np.minimum(w[i0:i1], 0.4)
    w[hard] = 0.0
    return starts, w


def stft_features(X, fs, short_s, hop_s, bands=((0.5, 4), (4, 8), (8, 13), (13, 30))):
    C, T = X.shape
    nper = int(round(short_s * fs));
    nover = nper - int(round(hop_s * fs))
    feats = []
    for c in range(C):
        f, t, Z = stft(X[c], fs=fs, nperseg=nper, noverlap=nover, boundary=None, padded=False)
        P = (Z.real ** 2 + Z.imag ** 2) + 1e-12  # (F, nF)
        total = np.trapz(P, f, axis=0)
        rel = [np.trapz(P[(f >= lo) & (f < hi)], f[(f >= lo) & (f < hi)], axis=0) / total for lo, hi in bands]
        rel = np.stack(rel, axis=1)  # (nF, n_bands)
        Pn = P / np.sum(P, axis=0, keepdims=True)
        ent = -(Pn * np.log(Pn)).sum(axis=0)  # (nF,)
        dpos = np.maximum(np.diff(P, axis=1), 0.0)
        flux = np.r_[0.0, np.sqrt((dpos ** 2).sum(axis=0))]  # (nF,)
        # time-domain on same framing
        win = nper;
        hop = nper - nover;
        starts = (np.arange(P.shape[1]) * hop).astype(int)
        frames = np.stack([X[c, s:s + win] for s in starts])
        rms = np.sqrt((frames ** 2).mean(axis=1))
        linelen = np.mean(np.abs(np.diff(frames, axis=1)), axis=1)
        tkeo = np.mean(np.abs(frames[:, 1:-1] ** 2 - frames[:, :-2] * frames[:, 2:]), axis=1)
        feats.append(np.c_[rel, ent, flux, rms, linelen, tkeo])  # (nF, n_feat)
    names = [f"rel_{lo}-{hi}" for lo, hi in bands] + ["ent", "flux", "rms", "ll", "tkeo"]
    return np.stack(feats, axis=0), names, starts  # (C, nF, n_feat), names, starts


def rolling_robust_z(A, win):
    # A: (C, nF, n_feat)
    C, N, F = A.shape
    Z = np.empty_like(A)
    for c in range(C):
        for ftr in range(F):
            x = A[c, :, ftr]
            med = np.array([np.median(x[max(0, i - win):i + 1]) for i in range(N)])
            mad = np.array([np.median(np.abs(x[max(0, i - win):i + 1] - med[i])) for i in range(N)]) + 1e-9
            Z[c, :, ftr] = (x - med) / (1.4826 * mad)
    return Z


def topk_mean(S, k_frac=0.4):
    k = max(1, int(round(k_frac * S.shape[0])))
    return np.sort(S, axis=0)[-k:].mean(axis=0)


def poisson_disk_select(times, score, min_gap_s, need_n, fs):
    keep_idx, last_end = [], -np.inf
    order = np.argsort(score)[::-1]
    gap = int(round(min_gap_s * fs))
    for i in order:
        t = times[i]
        if t >= last_end:
            keep_idx.append(i);
            last_end = t + gap
            if len(keep_idx) == need_n: break
    return sorted(keep_idx)


# ---- glue it together
import mne


def pick_short_segments(raw, SHORT=1.0, HOP=None, need_n=200):
    if HOP is None: HOP = SHORT / 4
    raw = preprocess_raw(raw)
    X, fs = raw.get_data(picks='eeg'), raw.info['sfreq']
    feats, names, starts = stft_features(X, fs, SHORT, HOP)
    Z = rolling_robust_z(feats, win=int(round(30 / HOP)))  # 30 s window
    # simple weights (tweakable)
    w = np.array([0.5, 0.4, 0.5, 0.4] + [-0.4, 1.5, 0.7, 1.0, 1.0])[:Z.shape[2]]
    S_ch = (Z * w).sum(axis=2)  # (C, nF)
    S = topk_mean(S_ch, 0.4)  # (nF,)
    _, wart = artifact_weights_from_mne(raw, SHORT, HOP)  # (nF,)
    S_eff = wart * S
    keep = poisson_disk_select(starts, S_eff, SHORT * 1.1, need_n, fs)
    # convert frame starts to sample indices
    return [int(starts[i]) for i in keep]
