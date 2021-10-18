'''Utility functions'''
import ast
from typing import Iterable, Tuple
from os import listdir
from os.path import isfile, isdir
from open3d import geometry

import pandas as pd
import numpy as np

def converter(instr):
    return np.asarray(ast.literal_eval(instr))

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

def compute_pca(mesh: geometry.TriangleMesh) -> Tuple[np.array, np.array, np.array, np.array]:
    '''Computes the eigenvalues and eigenvectors of a mesh

    Args:
        mesh (geometry.TriangleMesh): The mesh to compute the eigenvalues for

    Returns:
        Tuple[np.array, np.array]: A 4-tuple containing the eigenvectors on respectively the x, y, and z axis, and the eigenvalues
    '''
    vertex_count = len(mesh.vertices)

    x_coords = []
    y_coords = []
    z_coords = []

    for row in np.asarray(mesh.vertices):
        x_coords.append(row[0])
        y_coords.append(row[1])
        z_coords.append(row[2])

    #Fill in matrix with coordinate points
    A = np.zeros((3, vertex_count), dtype=float)
    A[0] = np.asarray(x_coords)
    A[1] = np.asarray(y_coords)
    A[2] = np.asarray(z_coords)

    #Calculate covariance matrix
    A_cov = np.cov(A)

    #Calculate eigens
    eigenvalues, eigenvectors = np.linalg.eig(A_cov)

    eigenvalues, eigenvectors = sort_eigen_vectors(eigenvalues, eigenvectors)
    
    x_axis = eigenvectors[0]
    y_axis = eigenvectors[1]
    z_axis = np.cross(eigenvectors[0], eigenvectors[1])
    
    return x_axis, y_axis, z_axis, eigenvalues

def grouped(iterable: Iterable, count: int) -> Tuple[Iterable]:
    '''Returns the iterable zipped into groups of a specified count

    Args:
        iterable (Iterable): The iterable to group
        count (int): The size of the groups (if available)

    Returns:
        Tuple[Iterable]: The count-tuple of iterable list
    '''
    return zip(*[iter(iterable)]*count)
