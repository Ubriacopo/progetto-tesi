import unittest

from data.media.audio import AudioCollector, AudioPreProcessingPipeline
from data.media.signal import SignalCollector, SignalMediaPreProcessingPipeline, extract_trial_data
from data.media.video import VideoCollector, VideoPreProcessingPipeline
from .text import TextCollector, TextProcessingPipeline


# TODO: Test all media types
class MediaTest(unittest.TestCase):
    def test_text(self):
        # Text seems ok
        collector = TextCollector([], TextProcessingPipeline.default())
        collector.load_additional_resource("./../resources/Text_Test.txt")
        collector.get_media(0)
        collector.load_additional_resource("./../resources/Text_Test.txt")
        collector.load_additional_resource("./../resources/Text_Test.txt")
        collector.get_media(2)

    def test_video(self):
        collector = VideoCollector(VideoPreProcessingPipeline())
        collector.load_additional_resource("./../resources/Exp1_P01_face/P1_4_face.mov")
        processed = collector.get_media(0)
        print(processed)

    def test_eeg(self):
        extract_trial_data("./../resources/", "./../resources/Data_Preprocessed_P01.mat")
        collector = SignalCollector.AMIGOS(SignalMediaPreProcessingPipeline())
        collector.load_additional_resource("./../resources/Data_Preprocessed_P01.npz")
        processed = collector.get_media(0)
        print(processed)

    def test_audio(self):
        collector = AudioCollector(AudioPreProcessingPipeline())
        collector.load_additional_resource("./../resources/Exp1_P01_face/P1_4_face.mov")
        processed = collector.get_media(0)
        print(processed)


if __name__ == '__main__':
    unittest.main()
