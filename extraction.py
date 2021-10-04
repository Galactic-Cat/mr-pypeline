'''Module for preprocessing the off and ply files'''
from logging import getLogger
from os import getcwd, listdir, mkdir
from os.path import basename, exists, isfile, isdir
from re import X, match, search, split
from typing import Dict
from open3d import geometry, io, utility
import numpy as np
import math

log = getLogger('feature_extraction')

def dist_between_points(vector_1: np.array, vector_2: np.array) -> float:
    if vector_1 is None or vector_2 is None:
        log.error('Cannot find distance between null points.')
        return None

    total = 0

    for i,j in zip(vector_1, vector_2):
        total += (i-j)**2

    dist = math.sqrt(total)

    return dist


if __name__ == '__main__':
    print(dist_between_points(np.asarray([0,0,0]), np.asarray([1,2,3])))