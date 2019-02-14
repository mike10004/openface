import testutils
import unittest2
import glob
import cli
import os


testutils.configure_logging()


def _build_extractor(max_cache_size=None):
    fileDir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
    modelDir = os.path.join(fileDir, 'models')
    dlibModelDir = os.path.join(modelDir, 'dlib')
    openfaceModelDir = os.path.join(modelDir, 'openface')
    dlibFacePredictorPath = os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat")
    openfaceModelPath = os.path.join(openfaceModelDir, 'nn4.small2.v1.t7')
    imgDim = 96
    return cli.Extractor(dlibFacePredictorPath, openfaceModelPath, imgDim, max_cache_size)


class TestExtractor(unittest2.TestCase):

    def test_extract(self):
        extractor = _build_extractor()
        image_paths = glob.glob(os.path.join(os.path.dirname(__file__), '..', 'images', 'examples', '*.jpg'))
        for image_path in image_paths:
            template = extractor.extract(image_path)
            self.assertIsNotNone(template, image_path)


class TestModuleMethods(unittest2.TestCase):

    def test__parse_cache_size(self):
        test_cases = {
            '50b': 50,
            '1': 1,
            '1m': 1024 * 1024,
            '1k': 1024 * 1,
            '5k': 1024 * 5,
            '5K': 1024 * 5,
            '10g': 1024 * 1024 * 1024 * 10,
            '2.5g': int(1024 * 1024 * 1024 * 2.5),
            '103': 103,
            '-1': None,
            '-100': None,
        }
        for token, expected in test_cases.items():
            actual = cli._parse_cache_size(token)
            with self.subTest():
                if actual is not None:
                    self.assertIsInstance(actual, int)
                self.assertEqual(expected, actual, token)