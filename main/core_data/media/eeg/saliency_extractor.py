from typing import Optional, Tuple

import mne
import numpy as np
from scipy.signal import stft


class EEGFeatureExtractor:
    epsilon: float = 1e-12

    def __init__(self, raw: mne.io.RawArray, weights: Optional[np.ndarray] = None):
        """
        Wanted weights are only for feats not for bands. These are passed at point extraction.
        :param raw:
        :param weights: Entropy - Spectral Flux - RMS - Line Length - TKEO (Teager-Kaiser)
        """
        self.raw = raw.copy().load_data()

        if weights is not None and weights.shape[0] != 5:
            raise ValueError(f"The shape of weights should be of 5 not {weights.shape[0]}")

        self.weights: Optional[np.ndarray] = weights
        if self.weights is None:
            # Entropy - Spectral Flux - RMS - Line Length - TKEO (Teager-Kaiser)
            self.weights = np.array([-0.4, 1.5, 0.7, -1.0, 1.0])

        self.k_frac = .4

    def process(self):
        # AMIGOS already does this.
        # self.raw.filter(l_freq, h_freq, phase='zero-double', verbose=False)
        # self.raw.notch_filter([notch, 2 * notch], phase='zero-double', verbose=False)
        # self.raw.set_eeg_reference('average')
        # Per-channel standardization (z-score)
        self.raw.apply_function(lambda x: (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + self.epsilon))

    def pick_segments(self, duration: float, hop: float,
                      bands: tuple[tuple, ...] = ((0.5, 4), (4, 8), (8, 13), (13, 30)),
                      band_weights: tuple[float, ...] = (0.5, 0.4, 0.5, 0.4),
                      need_number: int = 30):
        if len(bands) != len(band_weights):
            raise ValueError(f"Shape mismatch between bands and weights: {len(bands)}!= {len(band_weights)}")

        feats, names, starts = self._stft_features(duration, hop, bands)

        Z = self.rolling_robust_z(feats, window=int(round(30 / hop)))  # 30 s window
        # Heuristic: z-scores behave like this statistically in normal distribution: 99.7 % of data lie within [−3, 3].
        # This avoids  RMS, TEKO and LL to dominate saliecny score.
        Z = np.clip(Z, -6, 6)
        w = np.concatenate((band_weights, self.weights))[:Z.shape[2]]

        S_channels = (Z * w).sum(axis=2)
        k = max(1, int(round(self.k_frac * S_channels.shape[0])))
        S = np.sort(S_channels, axis=0)[-k:].mean(axis=0)
        S = np.nan_to_num(S, nan=-np.inf)

        _, wart = self.artifact_weights_from_mne(duration, hop)  # (nF,)
        S_eff = wart * S
        keep = self.poisson_disk_select(starts, S_eff, duration * 1.1, need_number, self.raw.info['sfreq'])
        # convert frame starts to sample indices
        return np.array([int(starts[i]) for i in keep])

    @staticmethod
    def poisson_disk_select(times: list, score: np.ndarray, min_gap_s: float, need_n: int, fs: int):
        keep_idx, keep_times, last_end = [], [], -np.inf
        order = np.argsort(score)[::-1]

        gap = int(round(min_gap_s * fs))
        for i in order:
            t = times[i]

            if all(abs(t - kt) >= gap for kt in keep_times):
                keep_idx.append(i)
                keep_times.append(t)

                if len(keep_idx) == need_n:
                    break

        return sorted(keep_idx, key=lambda x: times[x])

    def rolling_robust_z(self, a, window):
        """

        :param a: (C, nF, n_feat)
        :param window:
        :return:
        """
        C, N, F = a.shape

        window = max(1, int(window))
        Z = np.empty_like(a)

        for c in range(C):
            for ftr in range(F):
                x = a[c, :, ftr]
                med = np.array([np.median(x[max(0, i - window):i + 1]) for i in range(N)])
                mad = np.array([np.median(np.abs(x[max(0, i - window):i + 1] - med[i])) for i in range(N)])
                mad = mad + self.epsilon  # Ad a minimum term
                Z[c, :, ftr] = (x - med) / (1.4826 * mad)

        return Z

    # todo fix
    def artifact_weights_from_mne(self, duration_s, hop_s):
        annotations = []
        try:
            ann, _ = mne.preprocessing.annotate_amplitude(self.raw, flat=1e-6, peak=150e-6, verbose=False)
            annotations.append(ann)
        except Exception:
            pass

        try:
            a_muscle, _ = mne.preprocessing.annotate_muscle_zscore(
                self.raw, ch_type='eeg', threshold=4.0, filter_freq=(30., 45.), min_length_good=0.2, verbose=False
            )
            annotations.append(a_muscle)
        except Exception:
            pass

        fs = self.raw.info['sfreq']
        window = int(round(duration_s * fs))
        hop = int(round(hop_s * fs))
        nF = (1 + (self.raw.n_times - window) // hop) if self.raw.n_times >= window else 0
        if nF <= 0:
            # No frames so we can limit ourselves to returning empty
            return np.array([], dtype=int), np.array([], dtype=float)

        if annotations:
            A = annotations[0]
            for a in annotations[1:]:
                A += a
            self.raw.set_annotations(A)

        starts = np.arange(nF) * hop

        w = np.ones(nF)
        hard = np.zeros(nF, dtype=bool)

        for onset, duration, description in zip(
                self.raw.annotations.onset, self.raw.annotations.duration, self.raw.annotations.description
        ):
            s = int(round((onset - self.raw.first_time) * fs))
            e = s + int(round(duration * fs))
            i0 = max(0, (s - window) // hop + 1 if s > 0 else 0)
            i1 = min(nF, e // hop + 1)
            if 'BAD_flat' in description or 'BAD' in description and 'muscle' not in description:
                hard[i0:i1] = True
            if 'muscle' in description:
                w[i0:i1] = np.minimum(w[i0:i1], 0.4)

        w[hard] = 0.0
        return starts, w

    # Teager-Kaiser
    @staticmethod
    def tkeo(frames: np.ndarray):
        # Teager–Kaiser Energy Operator averaged per frame; captures bursty, high-frequency energy.
        # Sensitive to rapid local changes (popular in EEG, speech and seismic signal analysis)
        return np.mean(np.abs(frames[:, 1:-1] ** 2 - frames[:, :-2] * frames[:, 2:]), axis=1)

    @staticmethod
    def line_length(frames: np.ndarray):
        # > Line length (LL) is a simple but powerful time-domain EEG feature that measures the total amount of change in the signal within a window.
        # Line length per frame: average absolute first difference; proxies “spikiness/complexity”.
        # In EEG is usually correlates to high activity as scares/epilepsy. If not the case it measures artifacts.
        # TODO: Vedi come fare se voglio usare per artifacts.
        return np.mean(np.abs(np.diff(frames, axis=1)), axis=1)

    @staticmethod
    def rms(frames: np.ndarray):
        # Root-mean-square per frame
        # Measures the average signal amplitude (i.e., power or energy level) in a window — independent of frequency content.
        return np.sqrt((frames ** 2).mean(axis=1))

    @staticmethod
    def spectral_flux(magnitude: np.ndarray, eps: float):
        # Spectral flux measure of how quickly the power spectrum of a signal is changing
        # Spectral dynamics
        magnitude = magnitude / (magnitude.sum(axis=0, keepdims=True) + eps)
        positive_change = np.maximum(np.diff(magnitude, axis=1), 0.0)
        return np.r_[0.0, np.sqrt((positive_change ** 2).sum(axis=0))]

    @staticmethod
    def entropy(p: np.ndarray, eps: float):
        # Spectral entropy per frame (Shannon). High when spectrum is flat, low when concentrated.
        p_normalized = p / (np.sum(p, axis=0, keepdims=True) + eps)
        return -(p_normalized * np.log(p_normalized + eps)).sum(axis=0)

    def _stft_features(self, duration_s: float, hop_s: float, bands=((0.5, 4), (4, 8), (8, 13), (13, 30))):
        x, fs = self.raw.get_data(picks='eeg'), self.raw.info['sfreq']
        x = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + self.epsilon)
        C, T = x.shape

        n_samples = int(round(duration_s * fs))
        hop_samples = int(round(hop_s * fs))
        n_overlap = max(0, n_samples - max(1, hop_samples))

        if n_samples <= 0:
            raise ValueError("duration_s must be > 0")
        if hop_samples > n_samples and self.raw.n_times < n_samples + hop_samples:
            raise ValueError("hop_s too large for duration_s and signal length")

        f, t_sec, Z0 = stft(x[0], fs=fs, nperseg=n_samples, noverlap=n_overlap, boundary=None, padded=False)
        centers = np.round(t_sec * fs).astype(int)
        starts_all = np.clip(centers - n_samples // 2, 0, max(0, T - n_samples))
        nF = len(starts_all)

        feats = []
        for c in range(C):
            _, _, Z = stft(x[c], fs=fs, nperseg=n_samples, noverlap=n_overlap, boundary=None, padded=False)

            P = Z.real ** 2 + Z.imag ** 2
            nF_use = min(nF, P.shape[1])
            starts_use = starts_all[:nF_use]
            if nF_use <= 0:
                raise RuntimeError("No frames produced; check duration_s/hop_s vs signal length")

            P = P[:, :nF_use]
            total_P = np.trapezoid(P, f, axis=0) + self.epsilon
            relative_P = [
                np.trapezoid(P[(f >= lo) & (f < hi)], f[(f >= lo) & (f < hi)], axis=0) / total_P
                for lo, hi in bands
            ]

            # Time domain same framing
            frames = np.stack([x[c, s:s + n_samples] for s in starts_use])

            feats.append(
                np.c_[
                    np.stack(relative_P, axis=1),  # Relative bandpower per frame
                    self.entropy(P, self.epsilon), self.spectral_flux(np.sqrt(P), self.epsilon),
                        # Time-domain dynamics
                    self.rms(frames), self.line_length(frames), self.tkeo(frames)
                ])

        return (
            np.stack(feats, axis=0),
            [f"rel_{lo}-{hi}" for lo, hi in bands] + ["ent", "flux", "rms", "ll", "tkeo"],
            starts_all
        )
