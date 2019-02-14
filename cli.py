#!/usr/bin/env python2
#
# Example to compare the faces in two images.
# Brandon Amos
# 2015/09/29
#
# Copyright 2015-2016 Carnegie Mellon University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modifications (c) Mike Chaberski distributed under same license

import argparse
import cv2
import sys
import itertools
import os
import re
import numpy as np
import openface
import logging
import csv
import cachetools


_log = logging.getLogger(__name__)



class OpenfaceException(Exception):
    pass


# noinspection PyTypeChecker
def _create_arg_parser(fileDir):
    modelDir = os.path.join(fileDir, 'models')
    dlibModelDir = os.path.join(modelDir, 'dlib')
    openfaceModelDir = os.path.join(modelDir, 'openface')
    parser = argparse.ArgumentParser()

    parser.add_argument('imgs', type=str, nargs='+', help="Input images.")
    parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                        default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
    parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",
                        default=os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'))
    parser.add_argument('--imgDim', type=int,
                        help="Default image dimension.", default=96)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--cache-size', metavar="N", help="set max cache size")
    return parser


_FACTORS = {
    'b': 1024 ** 0,
    'k': 1024 ** 1,
    'm': 1024 ** 2,
    'g': 1024 ** 3,
    't': 1024 ** 4

}


def _parse_cache_size(size_str=None):
    if size_str is None:
        return None
    size_str = size_str.lower()
    m = re.match(r'^(-)?(\d+(?:\.\d+)?)([bkmgt]|kb|mb|gb|tb)?$', size_str)
    if not m:
        raise ValueError("invalid size string")
    neg = m.group(1)
    if neg:
        return None
    literal, factor_str = m.group(2), m.group(3)
    factor = 1
    if factor_str:
        factor = _FACTORS[factor_str[0]]
    return int(float(literal) * factor)




class Extractor(object):

    def __init__(self, dlibFacePredictorPath, networkModelPath, imgDim, max_cache_size=None):
        self.align = openface.AlignDlib(dlibFacePredictorPath)
        self.net = openface.TorchNeuralNet(networkModelPath, imgDim)
        self.cache = None
        self.imgDim = imgDim
        if max_cache_size is not None:
            self.cache = cachetools.LRUCache(max_cache_size)

    def _maybe_cache(self, key, value):
        if self.cache is not None:
            self.cache[key] = value

    def extract(self, imgPath):
        _log.debug("lookup %s", imgPath)
        if self.cache is not None:
            try:
                return self.cache[imgPath]
            except KeyError:
                pass
        _log.debug("extract %s", imgPath)
        bgrImg = cv2.imread(imgPath)
        if bgrImg is None:
            _log.info("Unable to load image: {}".format(imgPath))
            return None
        rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
        _log.debug("Original size: %s", rgbImg.shape)
        bb = self.align.getLargestFaceBoundingBox(rgbImg)
        if bb is None:
            _log.debug("Unable to find a face: %s", imgPath)
            self._maybe_cache(imgPath, None)
            return None
        alignedFace = self.align.align(self.imgDim, rgbImg, bb, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        if alignedFace is None:
            _log.debug("Unable to align image: %s", imgPath)
            self._maybe_cache(imgPath, None)
            return None
        rep = self.net.forward(alignedFace)
        _log.debug("extracted template of size %s", sys.getsizeof(rep))
        self._maybe_cache(imgPath, None)
        return rep


def main():
    np.set_printoptions(precision=2)
    fileDir = os.path.dirname(os.path.realpath(__file__))
    parser = _create_arg_parser(fileDir)
    args = parser.parse_args()
    extractor = Extractor(args.dlibFacePredictor, args.networkModel, args.imgDim, _parse_cache_size(args.cache_size))
    ofile = sys.stdout
    csvout = csv.writer(ofile)
    for (img1, img2) in itertools.combinations(args.imgs, 2):
        probe = extractor.extract(img1)
        gallery = None if probe is None else extractor.extract(img2)
        score = ''
        if probe is not None and gallery is not None:
            d = probe - gallery
            score = np.dot(d, d)
        csvout.writerow([score, img1, img2])


if __name__ == '__main__':
    exit(main())
