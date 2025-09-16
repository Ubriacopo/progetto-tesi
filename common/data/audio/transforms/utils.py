from common.data.audio import Audio


def check_audio_data(x, data_type: type):
    if not isinstance(x, Audio):
        raise TypeError("Given object is not of required type Audio")

    if x.data is None:
        raise ValueError("Audio has to be loaded first.")

    if not isinstance(x.data, data_type):
        raise TypeError("Given audio object is not valid")
