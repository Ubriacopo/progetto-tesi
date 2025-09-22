import mne
import numpy as np
from scipy.signal import stft


class EEGFeatureExtractor:
    epsilon: float = 1e-12

    def __init__(self, raw: mne.io.RawArray):
        self.raw = raw.copy().load_data()
        self.k_frac = .4

    def process(self):
        # AMIGOS already does this.
        # self.raw.filter(l_freq, h_freq, phase='zero-double', verbose=False)
        # self.raw.notch_filter([notch, 2 * notch], phase='zero-double', verbose=False)
        # self.raw.set_eeg_reference('average')

        # Per-channel standardization (z-score)
        self.raw.apply_function(lambda x: (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + self.epsilon))

    # todo revisiona. Forse ci basta l'altro su start at max ranges. Probabilmente scartalo
    def pick_intervals(self, lengths=(1.0, 2.0, 3.0, 4.0), bands: tuple[tuple] = ((0.5, 4), (4, 8), (8, 13), (13, 30)),
                       hop: float = 0.5, need_number: int = 10, gap_factor: float = 1.1):
        feats, names, starts = self._stft_features(duration_s=1.0, hop_s=hop, bands=bands)  # base=1s example
        Z = self.rolling_robust_z(feats, window=int(round(30 / hop)))

        w_feat = np.array([0.5, 0.4, 0.5, 0.4] + [-0.4, 1.5, 0.7, 1.0, 1.0])[:Z.shape[2]]
        S = (Z * w_feat).sum(axis=2)  # (C, nF)
        k = max(1, int(round(self.k_frac * S.shape[0])))
        S = np.sort(S, axis=0)[-k:].mean(axis=0)  # (nF,)
        _, wart = self.artifact_weights_from_mne(1.0, hop)
        S_eff = np.nan_to_num(S * wart, nan=-np.inf)  # (nF,)

        # prefix sums for fast block means
        pref = np.r_[0.0, np.cumsum(S_eff)]
        fs = self.raw.info['sfreq']
        frame = int(round(hop * fs))  # hop in samples
        nF = len(S_eff)

        # For each frame, pick best length
        best_len = np.zeros(nF, dtype=float)
        best_score = np.full(nF, -np.inf)
        for L in lengths:
            k = max(1, int(round(L / hop)))  # number of frames in this length
            if k <= 0: continue
            # block sums S[i:i+k] via prefix sums
            block_sum = pref[k:] - pref[:-k]  # length nF - k + 1
            block_mean = block_sum / k
            # update best at valid i
            valid = block_mean > best_score[:len(block_mean)]
            best_score[:len(block_mean)][valid] = block_mean[valid]
            best_len[:len(block_mean)][valid] = L

        # Greedy variable-length selection with symmetric gap
        gap_samples = lambda L: int(round(gap_factor * L * fs))
        order = np.argsort(best_score)[::-1]
        keep, keep_spans = [], []

        for i in order:
            L = best_len[i]
            if L <= 0: continue
            start_smp = int(starts[i])
            stop_smp = start_smp + int(round(L * fs))
            # symmetric spacing wrt other kept intervals
            g = gap_samples(L)
            conflict = False
            for a, b in keep_spans:
                if not (stop_smp + g <= a or b + g <= start_smp):
                    conflict = True
                    break
            if conflict: continue
            keep.append(i)
            keep_spans.append((start_smp, stop_smp))
            if len(keep) == need_number: break

        # return (start, stop) in samples, time-sorted
        out = sorted(keep_spans, key=lambda ab: ab[0])
        return out

    # Optional augmentation: with probability p (say 0.2), replace a 4 s clip by a 1–3 s centered sub-clip; pad+mask back to the 4 s token length so batches stay uniform.
    # todo if shorter just crop center. O semplicemente fissi a 4s
    def pick_segments(self, duration: float, hop: float,
                      bands: tuple[tuple] = ((0.5, 4), (4, 8), (8, 13), (13, 30)), need_number: int = 30):
        feats, names, starts = self._stft_features(duration, hop, bands)
        Z = self.rolling_robust_z(feats, window=int(round(30 / hop)))  # 30 s window
        # Simple weights (tweakable)
        w = np.array([0.5, 0.4, 0.5, 0.4] + [-0.4, 1.5, 0.7, 1.0, 1.0])[:Z.shape[2]]

        S_channels = (Z * w).sum(axis=2)
        k = max(1, int(round(self.k_frac * S_channels.shape[0])))
        S = np.sort(S_channels, axis=0)[-k:].mean(axis=0)
        S = np.nan_to_num(S, nan=-np.inf)

        _, wart = self.artifact_weights_from_mne(duration, hop)  # (nF,)
        S_eff = wart * S
        keep = self.poisson_disk_select(starts, S_eff, duration * 1.1, need_number, self.raw.info['sfreq'])
        # convert frame starts to sample indices
        return [int(starts[i]) for i in keep]

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

    def artifact_weights_from_mne(self, duration_s, hop_s):
        annotations = []
        try:
            ann, _ = mne.preprocessing.annotate_amplitude(self.raw, flat=1e-6, peak=150e-6)
            annotations.append(ann)
        except Exception:
            pass

        try:
            a_muscle, _ = mne.preprocessing.annotate_muscle_zscore(
                self.raw, ch_type='eeg', threshold=4.0, filter_freq=(30., 45.), min_length_good=0.2
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

    def _stft_features(self, duration_s: float, hop_s: float, bands=((0.5, 4), (4, 8), (8, 13), (13, 30))):
        x, fs = self.raw.get_data(picks='eeg'), self.raw.info['sfreq']
        C, T = x.shape

        n_samples = int(round(duration_s * fs))
        n_overlap = n_samples - int(round(hop_s * fs))
        if n_samples <= 0 or n_overlap <= 0:
            raise ValueError("Samples and overlap must be positive")

        f, t_sec, Z0 = stft(x[0], fs=fs, nperseg=n_samples, noverlap=n_overlap, boundary=None, padded=False)
        centers = np.round(t_sec * fs).astype(int)
        starts_all = np.clip(centers - n_samples // 2, 0, max(0, T - n_samples))
        nF = len(starts_all)

        feats = []
        for c in range(C):
            _, _, Z = stft(x[c], fs=fs, nperseg=n_samples, noverlap=n_overlap, boundary=None, padded=False)

            P = (Z.real ** 2 + Z.imag ** 2) + self.epsilon

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

            P_normalized = P / (np.sum(P, axis=0, keepdims=True) + self.epsilon)
            entropy = -(P_normalized * np.log(P_normalized + self.epsilon)).sum(axis=0)

            mag = np.sqrt(P)  # Magnitude
            mag = mag / (mag.sum(axis=0, keepdims=True) + self.epsilon)
            positive_change = np.maximum(np.diff(mag, axis=1), 0.0)
            spectral_flux = np.r_[0.0, np.sqrt((positive_change ** 2).sum(axis=0))]

            # Time domain same framing
            frames = np.stack([x[c, s:s + n_samples] for s in starts_use])
            rms = np.sqrt((frames ** 2).mean(axis=1))

            line_length = np.mean(np.abs(np.diff(frames, axis=1)), axis=1)
            # Tagger–Kaiser Energy Operator averaged per frame; captures bursty, high-frequency energy.
            tkeo = np.mean(np.abs(frames[:, 1:-1] ** 2 - frames[:, :-2] * frames[:, 2:]), axis=1)
            feats.append(
                np.c_[
                    np.stack(relative_P, axis=1),  # Relative bandpower per frame
                    entropy,  # Spectral entropy per frame (Shannon). High when spectrum is flat, low when concentrated.
                    spectral_flux,  # Spectral flux measure of how quickly the power spectrum of a signal is changing
                    rms,  # Root-mean-square per frame
                    line_length,  # Line length per frame: average absolute first difference; proxies “spikiness/complexity”.
                    tkeo  # Teager–Kaiser Energy Operator averaged per frame; captures bursty, high-frequency energy
                ])

        return (
            np.stack(feats, axis=0),
            [f"rel_{lo}-{hi}" for lo, hi in bands] + ["ent", "flux", "rms", "ll", "tkeo"],
            starts_all
        )
