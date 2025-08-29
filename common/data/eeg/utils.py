import mne


def save_raw_fif(raw: mne.io.BaseRaw, path: str):
    # compressed FIF to save disk space
    raw.save(path, overwrite=True)  # e.g., "subject01_raw.fif"


def load_raw_fif(path: str) -> mne.io.BaseRaw:
    return mne.io.read_raw_fif(path, preload=True, verbose=False)


# Example: iterate and save one .fif per entry
# raw = collection_to_raw(sample)
# save_raw_fif(raw, f"{sample.entry_id}_raw.fif")
import re


def find_segments(raw: mne.io.BaseRaw, pattern: str):
    """Return list of (onset_s, duration_s, description) whose description matches regex `pattern`."""
    segs = []
    for on, dur, desc in zip(raw.annotations.onset,
                             raw.annotations.duration,
                             raw.annotations.description):
        if re.search(pattern, desc):
            segs.append((float(on), float(dur), desc))
    return segs


def get_segment_raw(raw: mne.io.BaseRaw, onset_s: float, duration_s: float) -> mne.io.BaseRaw:
    """Return a cropped Raw view of just that segment."""
    return raw.copy().crop(tmin=onset_s, tmax=onset_s + duration_s)


if __name__ == "__main__":
    import mne

    # todo
    # raws: list[mne.io.Raw] already created (same channels/sfreq ideally)
    for i, raw in enumerate(raws):
        raw.set_annotations(
            mne.Annotations(onset=[0.0],
                            duration=[raw.times[-1]],
                            description=[f"RUN_{i}"])
        )

    big_raw = mne.concatenate_raws(raws, on_mismatch="ignore")  # handles annotation offsets

    # (optional) if the file could be huge, let MNE split on disk automatically
    big_raw.save("all_recordings_raw.fif", overwrite=True, split_size="2GB")

    # Example: get the segment for subject P40, video 18
    hits = find_segments(big_raw, r"subject=P40.*video=18")
    onset, dur, desc = hits[0]
    seg_raw = get_segment_raw(big_raw, onset, dur)

    # Access samples/channels (Volts). Pick channels if you want just EEG:
    picks = mne.pick_types(seg_raw.info, eeg=True)
    data = seg_raw.get_data(picks=picks)  # shape (n_channels, n_times)
    fs = seg_raw.info["sfreq"]
    ch_names = [seg_raw.ch_names[i] for i in picks]
