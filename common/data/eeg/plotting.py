import matplotlib.pyplot as plt
import numpy as np


def plot_time_series(series, ticks: int = 5, frequency: int = 128):
    plt.figure(figsize=(10, 3))

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.ylabel(u"Voltage (\u03bcV)", fontsize=25)
    print("With frequency", frequency, " we have that our series lasts: ", len(series) / frequency)

    step = len(series) // ticks

    plt.xticks(
        np.arange(0, len(series) + 1, step),
        [round(x / frequency, 1) for x in np.arange(0, len(series) + 1, step)]
    )

    plt.xlabel("Time (s)", fontsize=25)
    plt.plot(series)
    return plt
