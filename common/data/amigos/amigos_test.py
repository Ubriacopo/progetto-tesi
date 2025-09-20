import unittest

from common.data.amigos.loader import AmigosPointsLoader
from common.data.amigos.preprocessing import AmigosPreprocessorFactory


class TestAmigos(unittest.TestCase):

    def test_interleaved_pipeline(self):
        input_path = "./resources/AMIGOS/"
        output_path = "./resources/AMIGOS/processed-interleaved/"

        loader = AmigosPointsLoader(input_path)
        processor = AmigosPreprocessorFactory.interleaved(
            output_path,
            cbramod_weights_path="./dependencies/cbramod/pretrained_weights.pth"
        )

        processor.run(loader)

    def test_default_pipeline(self):
        input_path = "./resources/AMIGOS/"
        output_path = "./resources/AMIGOS/processed-default/"

        loader = AmigosPointsLoader(input_path)
        processor = AmigosPreprocessorFactory.default(output_path)
        processor.run(loader)


if __name__ == '__main__':
    unittest.main()
