"""
We limit ourselves on recognition of these formats for the data generated.
If a processor does not produce one of these extensions the dataset won't be able to handle it correctly.
"""


def audio_extensions():
    return [".wav", ".mp3"]


def video_extensions():
    return [".mp4", ".mkv", ".mov", ".avi"]


def text_extensions():
    return [".txt"]
