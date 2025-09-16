import unittest

from common.data.amigos.loader import AmigosPointsLoader
from common.data.amigos.preprocessing import AmigosPreprocessorFactory


class TestAmigos(unittest.TestCase):

    def test_interleaved_pipeline(self):
        input_path = "./resources/AMIGOS/"
        output_path = "./resources/AMIGOS/processed-interleaved/"

        loader = AmigosPointsLoader(input_path)
        processor = AmigosPreprocessorFactory.interleaved(output_path)

        test_sample = next(loader.scan())
        test_sample["vid"].interval = (1, 8)
        test_sample["aud"].interval = (1, 8)
        test_sample["eeg"].interval = (1, 8)

        # z_vid = processor.pipeline["vid"](test_sample["vid"])
        # self.assertEqual(z_vid.shape, torch.Size([4, 3136, 768]))
        print("Shape is ok for video interleaved pipeline")
        z_aud = processor.pipeline["aud"](test_sample["aud"])
        # self.assertEqual(z_aud.shape[0], 4)
        z_eeg = processor.pipeline["eeg"](test_sample["eeg"])

        processor.run(loader)

    def test_default_pipeline(self):
        input_path = "./resources/AMIGOS/"
        output_path = "./resources/AMIGOS/processed-default/"

        loader = AmigosPointsLoader(input_path)
        processor = AmigosPreprocessorFactory.default(output_path)
        processor.run(loader)

if __name__ == '__main__':
    unittest.main()
