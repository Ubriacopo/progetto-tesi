import mne
import re


def find_segment_by_descriptor(raw: mne.io.BaseRaw, segment_annotation_identifier: str):
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
