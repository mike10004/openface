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
import json
import cv2
import sys
import os
import re
import numpy as np
import openface
import logging
import csv
import cachetools
import itertools

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
    parser.add_argument('--cache-size', type=int, metavar="N", default=20000, help="set max number of items in cache")
    parser.add_argument('--output-dir', help="set output directory")
    return parser


def _mkdirp(dirname):
    try:
        os.makedirs(dirname)
    except Exception:
        if not os.path.isdir(dirname):
            raise


class Serialist(object):

    disable_create_dirs = False

    def serialize(self, thing, writable):
        raise NotImplementedError()

    def deserialize(self, readable):
        raise NotImplementedError()

    def serialize_to_disk(self, thing, pathname):
        dirname = os.path.dirname(pathname)
        if not self.disable_create_dirs and not os.path.exists(dirname):
            _mkdirp(dirname)
        with open(pathname, 'wb') as ofile:
            self.serialize(thing, ofile)

    def deserialize_from_disk(self, pathname):
        with open(pathname, 'rb') as ifile:
            return self.deserialize(ifile)


class NumpyAwareJSONEncoder(json.JSONEncoder):

    def default(self, obj):  # pylint: disable=E0202
        if isinstance(obj, np.ndarray):
            if obj.ndim == 1:
                return obj.tolist()
            else:
                return [self.default(obj[i]) for i in range(obj.shape[0])]
        return json.JSONEncoder.default(self, obj)


class JsonSerialist(Serialist):

    def serialize(self, data, ofile, **kwargs):
        return json.dump(data, ofile, cls=NumpyAwareJSONEncoder, **kwargs)

    def deserialize(self, ifile):
        return np.array(json.load(ifile))


# not thread safe
class LoadingCache(object):

    def __init__(self, cache_size, load):
        self.lru_cache = cachetools.LRUCache(cache_size)
        self.load = load

    def get(self, key):
        try:
            return self.lru_cache[key]
        except KeyError:
            pass
        value = self.load(key)
        self.put(key, value)
        return value


    def put(self, key, value):
        self.lru_cache[key] = value


_POSIX_NOT_FOUND = 2
_IDENTITY = lambda x: x

class DiskLoader(object):

    def __init__(self, serialist, key_to_pathname=_IDENTITY):
        assert callable(key_to_pathname), "`key_to_pathname` argument must be callable"
        self.key_to_pathname = key_to_pathname
        self.serialist = serialist

    def __call__(self, *args, **kwargs):
        if not args and 'key' not in kwargs:
            raise ValueError("exactly one argument (key) is required")
        if len(args) > 0:
            key = args[0]
        else:
            key = kwargs['key']
        pathname = self.key_to_pathname(key)
        return self.serialist.deserialize_from_disk(pathname)

    def __contains__(self, key):
        return os.path.isfile(key)


class Extractor(object):

    def __init__(self, dlibFacePredictorPath, networkModelPath, imgDim):
        self.align = openface.AlignDlib(dlibFacePredictorPath or _find_model('dlib', _MODEL_FILENAME_DLIB))
        self.net = openface.TorchNeuralNet(networkModelPath or _find_model('openface', _MODEL_FILENAME_OPENFACE), imgDim)
        self.imgDim = imgDim

    def extract(self, imgPath):
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


def _collect_pathnames(mode, positionals, index_file):
    if mode == _MODE_EXTRACT:
        for p in positionals:
            yield os.path.normpath(p)
        if index_file:
            with open(index_file, 'r') as ifile:
                for line in ifile:
                    yield os.path.normpath(line.rstrip())
    elif mode == _MODE_MATCH:
        for a, b in itertools.combinations(positionals, 2):
            yield os.path.normpath(a), os.path.normpath(b)
        if index_file:
            with open(index_file, 'r') as ifile:
                for line in ifile:
                    a, b = re.split(r'\s+', line.rstrip(), 1)
                    yield os.path.normpath(a), os.path.normpath(b)
    else:
        raise ValueError("illegal mode")


class Matcher(object):

    def __init__(self, cache):
        self.cache = cache

    def compare_files(self, probe, gallery):
        p_rep = self.cache.get(probe)
        g_rep = self.cache.get(gallery)
        d = p_rep - g_rep
        score = np.dot(d, d)
        return score


def main():
    np.set_printoptions(precision=2)
    parser = _create_arg_parser()
    args = parser.parse_args()
    serialist = JsonSerialist()
    ofile = sys.stdout
    csvout = csv.writer(ofile)
    input_generator = _collect_pathnames(args.mode, args.files, args.files_from)
    if args.mode == _MODE_EXTRACT:
        extractor = Extractor(args.dlibFacePredictor, args.networkModel, args.imgDim)
        for image_file in input_generator:
            rep = extractor.extract(image_file)
            if rep is not None:
                output_pathname = os.path.join(args.output_dir, os.path.basename(image_file) + '.ofr')
                output_dir = args.output_dir if args.output_dir else os.getcwd()
                _mkdirp(output_dir)
                serialist.serialize_to_disk(rep, output_pathname)
                extract_ok = 1
            else:
                extract_ok = 0
            csvout.writerow([extract_ok, os.path.basename(image_file)])
    elif args.mode == _MODE_MATCH:
        loader = DiskLoader(serialist)
        cache = LoadingCache(args.cache_size, loader)
        matcher = Matcher(cache)
        for p_file, g_file in input_generator:
            score = matcher.compare_files(p_file, g_file)
            csvout.writerow([score, p_file, g_file])
    return 0


if __name__ == '__main__':
    exit(main())
