import mne
import numpy as np
import torch
from scipy.signal import stft


class SaliencyExtractor:
    epsilon: float = 1e-12

    def __init__(self, raw: mne.io.RawArray):
        self.raw = raw.copy().load_data()

    def process(self):
        # AMIGOS already does this.
        # self.raw.filter(l_freq, h_freq, phase='zero-double', verbose=False)
        # self.raw.notch_filter([notch, 2 * notch], phase='zero-double', verbose=False)
        # self.raw.set_eeg_reference('average')

        # Per-channel standardization (z-score)
        self.raw.apply_function(lambda x: (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-9))

    def pick_segments(self, duration: float, hop: float):
        self._stft_features()

    def _stft_features(self, duration_s: float, hop_s: float, bands=((0.5, 4), (4, 8), (8, 13), (13, 30))):
        x, fs = self.raw.get_data(picks='eeg'), self.raw.info['sfreq']
        C, T = x.shape

        n_samples = int(round(duration_s * fs))
        n_overlap = n_samples - int(round(hop_s * fs))
        if n_samples <= 0 or n_overlap <= 0:
            raise ValueError("Samples and overlap must be positive")

        f, t_sec, Z0 = stft(x[0], fs=fs, nperseg=n_samples, noverlap=n_overlap, boundary=None, padded=False)
        starts = np.round(t_sec * fs).astype(int)  # shape (nF,)
        nF = len(starts)

        feats = []
        for c in range(C):
            _, _, Z = stft(x[c], fs=fs, nperseg=n_samples, noverlap=n_overlap, boundary=None, padded=False)

            P = (Z.real ** 2 + Z.imag ** 2) + self.epsilon
            nF_use = min(nF, P.shape[1])
            if nF_use != P.shape[1] or nF_use != nF:
                # Trim to the common length if STFT gave one extra/less column due to edge effects
                P = P[:, :nF_use]

            total_P = np.trapezoid(P, f, axis=0)
            relative_P = [
                np.trapezoid(P[(f >= lo) & (f < hi)], f[(f >= lo) & (f < hi)], axis=0) / total_P
                for lo, hi in bands
            ]

            P_normalized = P / np.sum(P, axis=0, keepdims=True)
            entropy = -(P_normalized * np.log(P_normalized)).sum(axis=0)

            positive_change = np.maximum(np.diff(P, axis=1), 0.)
            spectral_flux = np.r_[0.0, np.sqrt((positive_change ** 2).sum(axis=0))]

            # Time domain same framing
            local_window = n_samples
            local_hop = n_overlap - n_samples

            starts = np.arange(P.shape[[1]] * local_hop).astype(int)
            frames = np.stack([x[c, s:s + local_window] for s in starts])
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
                ])  # (nF, n_feat)

        return np.stack(feats, axis=0), [f"rel_{lo}-{hi}" for lo, hi in bands] + ["ent", "flux", "rms", "ll",
                                                                                  "tkeo"], starts
