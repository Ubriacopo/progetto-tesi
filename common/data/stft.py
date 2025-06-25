import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from scipy.stats import stats

import VATE.media


# And it's also valid for EEG, where low frequencies (delta, theta, alpha, etc.) are most relevant.
def compute_stft(x, fs: float, top_clip: int = -1, **kwargs):
    f, t, z_xx = signal.stft(x, fs, **kwargs)
    # f and z_xx are sorted in ascending order

    # Do clipping operation to discard high frequencies that may be noisy or irrelevant
    # (e.g., beyond physiological range).
    z_xx = np.abs(z_xx[:top_clip])  # Just take the magnitude
    f = f[:top_clip]

    if np.isnan(z_xx).any():
        raise ValueError('stft returned NaN')

    return f, t, z_xx


def normalize_stft_zscore(f, t, z_xx, **kwargs):
    clip = 5  # To handle boundary effects
    z_xx = z_xx[:, clip:-clip]
    return f, t[clip:-clip], stats.zscore(z_xx, axis=-1)


def plot(f, t, z_xx):
    plt.figure(figsize=(15, 3))
    g1 = plt.pcolormesh(t, f, z_xx, shading="gouraud", vmin=-3, vmax=5)

    cbar = plt.colorbar(g1)
    tick_font_size = 15
    cbar.ax.tick_params(labelsize=tick_font_size)
    cbar.ax.set_ylabel("Power (Arbitrary units)", fontsize=15)

    plt.xlabel("Time (s)", fontsize=20)
    plt.xticks(fontsize=20)

    plt.ylabel("Frequency (Hz)", fontsize=20)
    plt.yticks(fontsize=20)
    return plt
