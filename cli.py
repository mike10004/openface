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
#
#
# Modifications (c) Mike Chaberski distributed under same license
#

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


_MODEL_FILENAME_OPENFACE = "nn4.small2.v1.t7"
_MODEL_FILENAME_DLIB = "shape_predictor_68_face_landmarks.dat"


def _find_model(category, filename, cwd=os.getcwd()):
    dirnames = [
        os.path.join(cwd, category),
        os.path.join(os.path.dirname(__file__), 'models', category),
        os.path.join(os.getenv('HOME'), '.local', 'share', 'openface', 'models', category),
    ]
    candidates = map(lambda dirname: os.path.join(dirname, filename), dirnames)
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise OpenfaceException("could not find " + filename + "; searched these directories: " + str(dirnames))


_MODE_EXTRACT = 'extract'
_MODE_MATCH = 'match'


# noinspection PyTypeChecker
def _create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=(_MODE_EXTRACT, _MODE_MATCH), default=_MODE_MATCH, metavar='MODE', help="operating mode ('extract' or 'match')")
    parser.add_argument('files', type=str, nargs='*', help="input files (images for 'extract' mode, templates for 'match' mode)")
    parser.add_argument('--files-from', metavar='FILE', help="read input files from FILE; one per line for 'extract' mode, two per line for 'match'")
    parser.add_argument('--dlibFacePredictor', type=str, metavar='FILE', help="Path to dlib's face predictor.")
    parser.add_argument('--networkModel', type=str, metavar='FILE', help="Path to Torch network model.")
    parser.add_argument('--imgDim', type=int, metavar='N', help="Default image dimension.", default=96)
    parser.add_argument('--verbose', action='store_true', help="be verbose")
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


class Serialist(object):

    disable_create_dirs = False

    def serialize(self, thing, writable):
        raise NotImplementedError()

    def deserialize(self, readable):
        raise NotImplementedError()

    def serialize_to_disk(self, thing, pathname):
        dirname = os.path.dirname(pathname)
        if not self.disable_create_dirs and not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)
        with open(pathname, 'wb') as ofile:
            self.serialize(thing, ofile)

    def deserialize_from_disk(self, pathname):
        with open(pathname, 'rb') as ifile:
            return self.deserialize(ifile)


class JsonSerialist(Serialist):

    def serialize(self, thing, writable):
        raise NotImplementedError()

    def deserialize(self, readable):
        raise NotImplementedError()


# not thread safe
class CompositeCache(object):

    def __init__(self, caches):
        self.caches = caches

    def get(self, key, load):
        for cache in self.caches:
            try:
                return cache[key]
            except KeyError:
                pass
        value = load(key)
        self.put(key, value)
        return value


    def put(self, key, value):
        for cache in self.caches:
            cache[key] = value

    @classmethod
    def build(cls, mem_cache_size, key_to_pathname):
        caches = []
        if mem_cache_size:
            caches.append(cachetools.LRUCache(mem_cache_size))
        if key_to_pathname:
            caches.append(DiskCache(JsonSerialist(), key_to_pathname))
        return CompositeCache(caches)


_POSIX_NOT_FOUND = 2


class DiskCache(object):

    def __init__(self, serialist, key_to_pathname=lambda x: x):
        assert callable(key_to_pathname), "`key_to_pathname` argument must be callable"
        self.key_to_pathname = key_to_pathname
        self.serialist = serialist

    def __getitem__(self, key):
        pathname = self.key_to_pathname(key)
        try:
            return self.serialist.deserialize_from_disk(pathname)
        except IOError as e:
            if e.errno == _POSIX_NOT_FOUND:
                raise KeyError()
            raise

    def __setitem__(self, key, value):
        pathname = self.key_to_pathname(key)
        self.serialist.serialize_to_disk(value, pathname)

    def __iter__(self):
        raise TypeError("DiskCache is not an iterable mapping type")

    def __contains__(self, key):
        return os.path.isfile(key)


def create_key_to_pathname_function(suffix='', parent_dir=None, strip_prefix=None):
    if not parent_dir:
        parent_dir = os.getcwd()
    def key_to_pathname(key):
        relative_path = key
        if strip_prefix and len(key) > len(strip_prefix) and key.startswith(strip_prefix):
            relative_path = key[len(strip_prefix):]
        relative_path += suffix
        return os.path.join(parent_dir, relative_path)
    return key_to_pathname


class Extractor(object):

    def __init__(self, dlibFacePredictorPath, networkModelPath, imgDim, cache):
        self.align = openface.AlignDlib(dlibFacePredictorPath or _find_model('dlib', _MODEL_FILENAME_DLIB))
        self.net = openface.TorchNeuralNet(networkModelPath or _find_model('openface', _MODEL_FILENAME_OPENFACE), imgDim)
        self.cache = cache
        self.imgDim = imgDim

    def extract(self, imgPath_):
        def load(imgPath):
            _log.debug("extract %s", imgPath)
            bgrImg = cv2.imread(imgPath)
            if bgrImg is None:
                _log.info("Unable to load image: %s", imgPath)
                return None
            rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
            _log.debug("Original size: %s", rgbImg.shape)
            bb = self.align.getLargestFaceBoundingBox(rgbImg)
            if bb is None:
                _log.debug("Unable to find a face: %s", imgPath)
                return None
            alignedFace = self.align.align(self.imgDim, rgbImg, bb, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
            if alignedFace is None:
                _log.debug("Unable to align image: %s", imgPath)
                return None
            rep = self.net.forward(alignedFace)
            _log.debug("extracted template of size %s", sys.getsizeof(rep))
            return rep
        return self.cache.get(imgPath_, load)


def _collect_pathnames(specifiers):
    pathnames = []
    for specifier in specifiers:
        if specifier and specifier[0] == '@' and os.path.isfile(specifier):
            with open(specifier[1:], 'r') as ifile:
                pathnames += [p for p in ifile.read().split(os.linesep) if p.strip() and not p[0] == '#']
            continue
        pathnames.append(specifier)
    return pathnames


def main():
    np.set_printoptions(precision=2)
    parser = _create_arg_parser()
    args = parser.parse_args()
    cache = CompositeCache.build(args.cache_size, args.template_storage_dir)
    if args.mode == _MODE_EXTRACT:
        extractor = Extractor(args.dlibFacePredictor, args.networkModel, args.imgDim, cache)
    ofile = sys.stdout
    csvout = csv.writer(ofile)
    image_pathnames = _collect_pathnames(args.imgs)
    for (img1, img2) in itertools.combinations(image_pathnames, 2):
        probe = extractor.extract(img1)
        gallery = None if probe is None else extractor.extract(img2)
        score = ''
        if probe is not None and gallery is not None:
            d = probe - gallery
            score = np.dot(d, d)
        csvout.writerow([score, img1, img2])


if __name__ == '__main__':
    exit(main())
