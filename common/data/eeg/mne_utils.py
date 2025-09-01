import mne
import re


# TODO ristruttura per usabilità
# todo non mi servono questi 2 metodi ma almeno non devo pensare a come si chiamano
def save_raw_fif(raw: mne.io.BaseRaw, path: str):
    # Compressed FIF to save disk space.
    raw.save(path, overwrite=True)  # e.g., "subject01_raw.fif"


def load_raw_fif(path: str) -> mne.io.BaseRaw:
    return mne.io.read_raw_fif(path, preload=True, verbose=False)


# todo use dataclasses?
def find_segment_by_descriptor(raw: mne.io.BaseRaw, segment_annotation_identifier):
    ann = raw.annotations

    return [
        (description, onset, duration)
        for description, onset, duration in zip(ann.description, ann.onset, ann.duration)
        if segment_annotation_identifier == description
        # if re.search(segment_annotation_identifier, description) We want 100% match no overlaps
    ]


def find_segments(raw: mne.io.BaseRaw, pattern: str):
    """
    Return list of (onset_s, duration_s, description) whose description matches regex `pattern`

    :param raw:
    :param pattern:
    :return:
    """
    segments = []
    annotations = raw.annotations
    for on, dur, desc in zip(annotations.onset, annotations.duration, annotations.description):
        if re.search(pattern, desc):
            segments.append((float(on), float(dur), desc))

    return segments


def get_segment_raw(raw: mne.io.BaseRaw, onset_s: float, duration_s: float) -> mne.io.BaseRaw:
    """
    Return a cropped Raw view of just that segment

    :param raw:
    :param onset_s:
    :param duration_s:
    :return:
    """
    return raw.copy().crop(tmin=onset_s, tmax=onset_s + duration_s)


# todo non serve una fn
def get_eeg_data(raw: mne.io.BaseRaw):
    picks = mne.pick_types(raw.info, eeg=True)
    return raw.get_data(picks=picks)  # shape (n_channels, n_times)
