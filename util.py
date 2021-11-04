'''Utility functions'''

from typing import Iterable, Tuple
from os import listdir
from os.path import isfile, isdir

import trimesh as tm
import numpy as np

BIN_COUNT = 30
SAMPLE_SIZE = 100000
SIZE_PARAM = 20000

def fix_mesh(mesh: tm.base.Trimesh) -> tm.base.Trimesh:
    tm.repair.fill_holes(mesh)
    return mesh

def calculate_mesh_center(mesh: tm.Trimesh):
    
    if mesh.is_watertight:
        return mesh.center_mass

    return mesh.centroid


def locate_mesh_files(input_path: str):

    # Setup input search
    folders = []
    files = []
    
    files.append(input_path) if isfile(input_path) else folders.append(input_path)

    # DFS the filesystem for .off and .ply files
    while len(folders) > 0:
        current_folder = folders.pop()

        for item in listdir(current_folder):
            item_path = current_folder + '/' + item

            if isdir(item_path):
                folders.append(item_path)
            elif isfile(item_path) and item_path[-4:] in ['.ply', '.off']:
                files.append(item_path)
    
    return files

def sort_eigen_vectors(eigen_values: np.array, eigen_vectors: np.array):
    eigen_values = np.abs(eigen_values)
    eigen_values, eigen_vectors = zip(*sorted(zip(eigen_values,eigen_vectors), reverse=True))
    eigen_vectors = np.asarray(eigen_vectors)

    return eigen_values, eigen_vectors


def grouped(iterable: Iterable, count: int) -> Tuple[Iterable]:
    '''Returns the iterable zipped into groups of a specified count

    Args:
        iterable (Iterable): The iterable to group
        count (int): The size of the groups (if available)

    Returns:
        Tuple[Iterable]: The count-tuple of iterable list
    '''
    return zip(*[iter(iterable)]*count)
