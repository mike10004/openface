import testutils
import unittest2
import glob
import cli
import os


testutils.configure_logging()


def _build_extractor():
    fileDir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
    modelDir = os.path.join(fileDir, 'models')
    dlibModelDir = os.path.join(modelDir, 'dlib')
    openfaceModelDir = os.path.join(modelDir, 'openface')
    dlibFacePredictorPath = os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat")
    openfaceModelPath = os.path.join(openfaceModelDir, 'nn4.small2.v1.t7')
    imgDim = 96
    return cli.Extractor(dlibFacePredictorPath, openfaceModelPath, imgDim)


class TestExtractor(unittest2.TestCase):

    def test_extract(self):
        extractor = _build_extractor()
        image_paths = glob.glob(os.path.join(os.path.dirname(__file__), '..', 'images', 'examples', '*.jpg'))
        for image_path in image_paths:
            template = extractor.extract(image_path)
            self.assertIsNotNone(template, image_path)


