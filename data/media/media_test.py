import unittest

from data.media.text import Text


# TODO: Test all media types
class MediaTest(unittest.TestCase):
    def test_text(self):
        media = Text("../../resources/Text_Test.txt")
        media.load()
        media.process()
        print(media.processed_text)

        self.assertEqual(True, False)  # add assertion here

    def test_video(self):
        pass

    def test_eeg(self):
        pass

    def test_audio(self):
        pass


if __name__ == '__main__':
    unittest.main()
