from abc import abstractmethod, ABC

import numpy as np
import torch

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


class RandomLogUniformIntervalsSegmenter(Segmenter):
    SHORT_LENGTH = 4
    MEDIUM_LENGTH = 12
    LONG_LENGTH = 32

    def __init__(self, min_length: int, max_length: int, num_segments: int,
                 jitter_frac: dict = None,
                 coverage_resolution_sec: float = 0.1, coverage_cap_k: int = 5,
                 same_iou_max=None, cross_iou_max: float = 0.7):
        super().__init__(max_length)

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

    def compute_segments(self, sample: EEG) -> list[tuple[int, int]]:
        buckets: list[tuple[float, float, str, float]] = []
        t = sample.data.duration
        medium_anchors, long_anchors = [], []
        # Coverage tracker in seconds (no index math)
        num_slots = int(np.ceil(sample.data.duration / self.coverage_resolution_sec))
        medium_coverage, long_coverage = np.zeros(num_slots, dtype=np.int16), np.zeros(num_slots, dtype=np.int16)

        durations = [self.sample_duration_log_uniform() for _ in range(self.num_segments)]
        for duration in durations:
            if duration < self.SHORT_LENGTH:
                # adds to buckets but also to anchors todo
                self.extract_anchor(
                    t=t, d=duration,
                    segments=buckets,
                    extraction_jitter=.1,
                    coverage=medium_coverage
                )
            elif duration < self.MEDIUM_LENGTH:
                self.extract(
                    t=t, d=duration,
                    anchors=medium_anchors,
                    segments=buckets,
                    extraction_jitter=.1,
                    coverage=medium_coverage,
                )
            elif duration < self.LONG_LENGTH:
                self.extract(
                    t=t, d=duration,
                    anchors=long_anchors,
                    segments=buckets,
                    extraction_jitter=.1,
                    coverage=long_coverage,
                )

        # todo remap buckets
        return buckets

    def extract_anchor(self,
                       t: float, d: float,
                       segments: list[tuple[float, float, str, float]],
                       extraction_jitter: float,
                       coverage: np.ndarray,  # todo non serve coverage qui basta controlli
                       attempt: int = 0
                       ) -> bool:
        # Try on saliency or on random TODO
        pass

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

    # todo param retries -> si deve arrendere. mi sa che coincide con long
    def extract(self,
                # t is media time, d is extraction duration
                t: float, d: float,
                # todo add index of segments for anchor
                anchors: list[tuple[float, float, int]],
                # (start, stop, type, duration)
                segments: list[tuple[float, float, str, float]],
                extraction_jitter: float,
                coverage: np.ndarray,
                attempt: int = 0
                ):
        """

        :param t:
        :param d:
        :param anchors:
        :param segments:
        :param extraction_jitter:
        :param coverage:
        :param attempt:
        :return:
        """
        if attempt > self.max_attempts:
            return False  # Could not extract

        # At chance center on short + jitter or just sample randomly
        jitter_frac = self.jitter_frac["M"]

        anchor_appended = False
        center_on_anchor = np.random.random() < .5
        if center_on_anchor and len(anchors) != 0:
            shorts = [(s, e, idx) for idx, (s, e, b, _) in enumerate(segments) if b == 'S']
            anchor_idx = np.random.randint(0, len(shorts))
            # todo mi manca index
            if len(list(filter(lambda x: x[-1] == anchor_idx, anchors))) != 0:
                return self.extract(t, d, anchors, segments, extraction_jitter, coverage, attempt + 1)

            anchor_start, anchor_end, anchor_idx = shorts[anchor_idx]
            center = (anchor_start + anchor_end) / 2 + np.random.uniform(-jitter_frac, jitter_frac) * d
            start = np.clip(center - d / 2, 0., max(0., t - d))

            anchors.append(shorts[anchor_idx])
            anchor_appended = True  # To undo in case I have to do better
        else:
            start = np.random.uniform(0., max(1e-9, t - d))
        # Calculate end point
        stop = start + d

        if not self.ok_iou(start, stop, "M", segments) or not self.check_coverage(start, stop, coverage):
            # Already exists so we have to retry extracting. TODO
            if anchor_appended:
                anchors.pop()
            return self.extract(t, d, anchors, segments, extraction_jitter, coverage, attempt + 1)

        # Tiny jitter to avoid identical cuts.
        if extraction_jitter > 0:
            jitter = (np.random.uniform(-1, 1) * extraction_jitter) * d
            start = float(np.clip(start + jitter, 0., max(0., t - d)))
            stop = start + d

        segments.append((start, stop, "M", d))
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


class EEGFeatureIntervalSegmenter(Segmenter):
    def compute_segments(self, sample: EEG) -> list[tuple[int, int]]:
        pass  # TODO

# todo revisiona e studia e fai
import numpy as np
import mne
from scipy.signal import welch
from scipy.ndimage import gaussian_filter1d

def compute_bandpower(psd, freqs, fmin, fmax):
    """Integrate PSD between fmin-fmax."""
    idx = np.logical_and(freqs >= fmin, freqs < fmax)
    return np.trapz(psd[..., idx], freqs[idx], axis=-1)

def trimmed_mean(x, trim=0.1, axis=0):
    """Robust average across channels to suppress outliers."""
    x_sorted = np.sort(x, axis=axis)
    n = x.shape[axis]
    lo = int(np.floor(trim * n))
    hi = int(np.ceil((1.0 - trim) * n))
    slicer = [slice(None)] * x.ndim
    slicer[axis] = slice(lo, hi)
    return x_sorted[tuple(slicer)].mean(axis=axis)

def saliency_from_raw(
    raw: mne.io.BaseRaw,
    l_freq=0.5, h_freq=45.0, notch=50.0,
    win_sec=0.256, hop_sec=0.1,
    bands=((1,4), (4,8), (8,13), (13,30)),    # δ, θ, α, β
    flux_weight=1.0, ll_weight=0.5, band_weight=1.0,
    smooth_sigma=1.0,                          # in frames (Gaussian sigma)
    refractory_sec=1.0,                        # min distance between peaks
    peak_prom_z=1.0,                           # min z-prominence for peaks
    picks="eeg"
):
    """
    Returns:
        times: (F,) times (sec) at frame centers
        saliency: (F,) standardized saliency score
        peak_times: (P,) seconds of saliency peaks (anchors)
    """
    raw = raw.copy().load_data()

    # --- Preprocess ---
    if notch:
        raw.notch_filter(freqs=[notch], picks=picks)
    raw.filter(l_freq=l_freq, h_freq=h_freq, picks=picks, method="fir", phase="zero-double", verbose=False)
    raw.set_eeg_reference("average")
    data = raw.get_data(picks=picks)  # (n_channels, n_samples)
    sfreq = raw.info["sfreq"]
    n_ch, n_samp = data.shape

    # z-score per channel
    data = (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-8)

    # --- Framing ---
    win = int(round(win_sec * sfreq))
    hop = int(round(hop_sec * sfreq))
    if win < 8:
        raise ValueError("win_sec too small for PSD")
    starts = np.arange(0, n_samp - win + 1, hop, dtype=int)
    centers = starts + win // 2
    times = centers / sfreq

    # Welch settings
    # For short frames, use nperseg=win with no overlap to keep compute simple
    nperseg = win
    noverlap = 0
    # Pre-allocate feature arrays
    F = len(starts)
    n_bands = len(bands)
    bandpow = np.zeros((n_ch, F, n_bands), dtype=np.float32)
    line_len = np.zeros((n_ch, F), dtype=np.float32)

    # --- Feature extraction per frame ---
    for j, s0 in enumerate(starts):
        s1 = s0 + win
        seg = data[:, s0:s1]  # (n_ch, win)

        # PSD per channel
        freqs, psd = welch(seg, fs=sfreq, nperseg=nperseg, noverlap=noverlap, axis=-1)
        # bandpowers
        for bidx, (fmin, fmax) in enumerate(bands):
            bandpow[:, j, bidx] = compute_bandpower(psd, freqs, fmin, fmax)

        # line length (per channel)
        diff = np.abs(np.diff(seg, axis=-1))
        line_len[:, j] = diff.mean(axis=-1)

    # --- Normalize features over time (per channel) ---
    def zscore(x, axis=-1, eps=1e-8):
        mu = x.mean(axis=axis, keepdims=True)
        sd = x.std(axis=axis, keepdims=True)
        return (x - mu) / (sd + eps)

    bandpow_z = zscore(bandpow, axis=1)   # (ch, F, bands)
    ll_z = zscore(line_len, axis=1)       # (ch, F)

    # --- Spectral flux (frame-to-frame change in bandpower) ---
    flux = np.zeros_like(bandpow_z)
    flux[:, 1:, :] = np.maximum(0.0, bandpow_z[:, 1:, :] - bandpow_z[:, :-1, :])  # positive changes
    # Aggregate features across bands & channels
    band_feat = trimmed_mean(bandpow_z.mean(axis=-1), trim=0.1, axis=0)     # (F,)
    flux_feat = trimmed_mean(flux.mean(axis=-1), trim=0.1, axis=0)          # (F,)
    ll_feat   = trimmed_mean(ll_z, trim=0.1, axis=0)                         # (F,)

    # --- Combined saliency ---
    sal = band_weight * band_feat + flux_weight * flux_feat + ll_weight * ll_feat
    # Smooth a bit to reduce spikiness (sigma in frames)
    if smooth_sigma and smooth_sigma > 0:
        sal = gaussian_filter1d(sal, sigma=smooth_sigma, mode="nearest")

    # Final z-score
    sal = (sal - sal.mean()) / (sal.std() + 1e-8)

    # --- Peak picking (non-maximum suppression + refractory) ---
    # Simple prominence: sal must exceed its local median by peak_prom_z*std
    peaks = []
    last_t = -np.inf
    refr_frames = int(round(refractory_sec / hop_sec))
    for i in range(1, F-1):
        if sal[i] > sal[i-1] and sal[i] > sal[i+1] and sal[i] >= peak_prom_z:
            if (i - (peaks[-1] if peaks else -10**9)) >= refr_frames:
                peaks.append(i)

    peak_times = times[peaks]
    return times, sal, peak_times
