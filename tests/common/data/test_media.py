import unittest

from VATE.preprocessing import SignalMediaPreProcessingPipeline, VideoPreProcessingPipeline, AudioPreProcessingPipeline, \
    TextProcessingPipeline
from common.data.media import NumpyDataMediaCollector, FileReferenceMediaCollector, PandasCsvDataMediaCollector
from utils.data import extract_trial_data


# TODO: Test all media types
class AMIGOSTest(unittest.TestCase):
    def test_eeg(self):
        extract_trial_data("../../../resources/", "./../resources/Data_Preprocessed_P01.mat")
        collector = NumpyDataMediaCollector([], SignalMediaPreProcessingPipeline())
        collector.load_additional_resource("./../resources/Data_Preprocessed_P01.npz")
        processed = collector.get_media(0)
        print(processed)

    def test_video(self):
        collector = FileReferenceMediaCollector(VideoPreProcessingPipeline())
        collector.load_additional_resource("./../resources/Exp1_P01_face/P1_4_face.mov")
        processed = collector.get_media(0)
        print(processed)

    def test_audio(self):
        collector = FileReferenceMediaCollector(AudioPreProcessingPipeline())
        collector.load_additional_resource("./../resources/Exp1_P01_face/P1_4_face.mov")
        processed = collector.get_media(0)
        print(processed)

    def test_text(self):
        # Text seems ok
        collector = PandasCsvDataMediaCollector([], TextProcessingPipeline.default())
        collector.load_additional_resource("./../resources/Text_Test.txt")
        collector.get_media(0)
        collector.load_additional_resource("./../resources/Text_Test.txt")
        collector.load_additional_resource("./../resources/Text_Test.txt")
        collector.get_media(2)


if __name__ == '__main__':
    unittest.main()
