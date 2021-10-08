'''Module for extracting features from 3D meshes'''
from logging import getLogger
from math import floor, pi, sqrt
from typing import List

import numpy as np
from open3d import geometry, io

from util import compute_pca, grouped

log = getLogger('features')
SAMPLE_SIZE = 250

def angle_between_randoms(mesh: geometry.TriangleMesh, samples: int = SAMPLE_SIZE) -> List[float]:
    '''Calculates angle between 3 random vertices.
    
    Args:
        mesh (geometry.TriangleMesh): The mesh from which we select the points
        num_of_points (int): Number of points to calculate
    
    Returns:
        List[float]: The list of angles between 3 random vertices
    '''
    vertices = np.asarray(mesh.vertices)
    vertex_count = vertices.shape[0] - (vertices.shape[0] % 3)
    entries = []

    for point_a, point_b, point_c in grouped(np.random.choice(vertices.shape[0], min(samples * 3, vertex_count), False), 3):
        ba_vector = point_a - point_b
        bc_vector = point_c - point_c
        cos_angle = np.dot(ba_vector, bc_vector) / (np.linalg.norm(ba_vector) * np.linalg.norm(bc_vector))

        # Normalize angle to a value in range [0, 1), for use in histograms
        entries.append(np.degrees(np.arccos(cos_angle)) / 360)

    return entries

def convert_entries_into_hist(entries: List[float], bin_count: int) -> np.array:
    '''Creates a histogram for a set of points.
    
    Args:
        entries(List[float]): List of the points to convert to a distribution.
    
    Returns:
        np.array: A numpy array representing the generated histogram
    '''
    step_size = 1.0 / bin_count
    histogram = np.zeros(bin_count, int)

    for entry in entries:
        index = floor(entry / step_size) # NOTE: 0.1 in python is slightly bigger than 0.1, (such that 1 // 0.1 = 9)
        histogram[index] += 1

    return histogram

def distance_between_points(vector_1: np.array, vector_2: np.array) -> float:
    '''Calculates distance between two points
    
    Args:
        vector_1 (np.array): Point one
        vector_2 (np.array): Point two
    
    Returns:
        float: The distance between the two points
    '''
    if vector_1 is None or vector_2 is None:
        log.error('Cannot find distance between null points.')
        return None

    return sqrt(sum((vector_1 - vector_2) ** 2))

def distance_barycenter_to_random(mesh: geometry.TriangleMesh, samples: int = SAMPLE_SIZE) -> List[float]:
    '''Calculates distance from barycenter to random vertices
    
    Args:
        mesh (geometry.TriangleMesh): The mesh from which we select the points
        samples (int): Number of points to calculate distance for
    
    Returns:
        List[float]: The distances between the barycenter of the mesh and random vertices
    '''
    barycenter = mesh.get_center()
    vertices = np.asarray(mesh.vertices)
    entries = []

    for index in list(np.random.choice(vertices.shape[0], min(samples, vertices.shape[0]), False)):
        entries.append(distance_between_points(barycenter, vertices[index]))
    
    return entries

def distance_random_to_random(mesh: geometry.TriangleMesh, samples: int = SAMPLE_SIZE) -> List[float]:
    '''Calculates distance from a random vertex to another random vertex.
    
    Args:
        mesh (geometry.TriangleMesh): The mesh from which we select the vertices
        samples (int): Number of point pairs to calculate distance for
    
    Returns:
        List[float]: The distance between 2 random vertices
    '''
    vertices = np.asarray(mesh.vertices)
    vertex_count = vertices.shape[0] - (vertices.shape[0] % 2)
    entries = []

    for index_a, index_b in grouped(np.random.choice(vertices, min(samples * 2, vertex_count), False), 2):
        entries.append(distance_between_points(vertices[index_a], vertices[index_b]))

    return entries

def simple_features(mesh: geometry.TriangleMesh) -> List[float]:
    '''Extract some simple features from a 3D mesh

    Args:
        mesh (geometry.TriangleMesh): The mesh to extract features from

    Returns:
        List[float]: A list of features, in order: surface area, compactness, AABB volume, diameter, and eccentricity
    '''
    values = []

    # Get the surface area
    surface_area = mesh.get_surface_area()
    values.append(surface_area)
    
    # Get compactness
    if mesh.is_watertight():
        values.append((surface_area ** 3) / (36 * pi * (mesh.get_volume() ** 2)))
    else:
        values.append(None)
        log.error("Cannot find compactness of a non-watertight mesh")
    
    # Get AABB volume
    values.append(mesh.get_axis_aligned_bounding_box().volume())

    # Get diameter
    # This is using a bruteforce method, but relying mostly on numpy's C code, so it takes ~0.5 seconds for 2000 vertices
    vertices = np.asarray(mesh.vertices)
    vertex_count = vertices.shape[0]
    squared_distances = np.zeros((vertex_count, vertex_count))

    for i in range(vertex_count):
        squared_distances[i,:] = np.sum(np.square(vertices[i,:] - vertices), axis=1)

    values.append(np.sqrt(np.max(squared_distances)))
    
    # Get eccentricity
    _, _, _, eigenvalues = compute_pca(mesh)
    values.append(abs(max(eigenvalues)) / abs(min(eigenvalues)))
    
    return values

if __name__ == '__main__':
    mesh = io.read_triangle_mesh('./data_out/database/m100.off')
    mesh = mesh if not mesh.is_empty() else io.read_triangle_mesh('./data_out/m100.off') # NOTE: My data_out looks like this ~Simon

    simple_features(mesh)

    #lister = distance_barycenter_to_random(mesh, 10)
    #print(lister)

    #convert_entries_into_hist([0,.1,.2,.3,.4,.5,.6,.7,.8,.9], 10)
