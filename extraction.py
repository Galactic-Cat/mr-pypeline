'''Module for extracting features from 3D meshes'''
from logging import getLogger
from math import floor, pi, sqrt
from typing import List

from open3d import geometry, io
import numpy as np

import matplotlib.pyplot as plt

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

    # for point_a, point_b, point_c in grouped(np.random.choice(vertices.shape[0], False), 3):
    #     print(point_a, point_b, point_c)
    #     ba_vector = point_a - point_b
    #     bc_vector = point_c - point_c
    #     cos_angle = np.dot(ba_vector, bc_vector) / (np.linalg.norm(ba_vector) * np.linalg.norm(bc_vector))

    #     # Normalize angle to a value in range [0, 1), for use in histograms
    #     entries.append(np.degrees(np.arccos(cos_angle)))# / 360)

    for point in range(samples):
        random_indeces = np.random.choice(samples, size=3, replace= False)
        random_vertices = vertices[random_indeces, :]

        vector_b_to_a = random_vertices[0] - random_vertices[1]
        vector_b_to_c = random_vertices[2] - random_vertices[1]

        cos_angle = np.dot(vector_b_to_a, vector_b_to_c) / (np.linalg.norm(vector_b_to_a) * np.linalg.norm(vector_b_to_c))
        angle = np.arccos(cos_angle)

        # This normalizes the angles to values between 0 - 1, facilitates histogram creation.
        f_angle = np.degrees(angle)/360

        entries.append(f_angle)

    return entries

    return entries

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

    total = 0
    for i,j in zip(vector_1, vector_2):
        total += (i-j)**2
    dist = sqrt(total)

    return dist

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

    for point in range(samples):
        random_indeces = np.random.choice(vertices.shape[0], size=1, replace= False)
        random_vertex = vertices[random_indeces, :]
        dist = distance_between_points(barycenter, random_vertex[0])
        entries.append(dist)
    
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

    for point in range(samples):
        random_indeces = np.random.choice(vertices.shape[0], size=2, replace= False)
        random_vertices = vertices[random_indeces, :]
        dist = distance_between_points(random_vertices[0], random_vertices[1])
        entries.append(dist)

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

def normalize_features(entries: List[float]) -> List[float]:

    total = sum(entries)

    entries = [entry/total for entry in entries]

    return entries

def visualize_histogram(hist:np.array, title: str, output_path: str) -> None:

    fig = plt.hist(hist)
    plt.title(title)
    plt.savefig(output_path + title +'.png')

    return

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
    values.append((surface_area ** 3) / (36 * pi * (mesh.get_volume() ** 2)))
    
    # Get AABB volume
    values.append(mesh.get_axis_aligned_bounding_box().volume())
    
    # Get diameter
    # TODO: Check if this is always correct, in my mind it is, but I'm not sure
    oobb = mesh.get_oriented_bounding_box()
    values.append(max(oobb.get_max_bound() - oobb.get_min_bound()))
    
    # Get eccentricity
    eigenvalues, _ = compute_pca(mesh)
    values.append(abs(max(eigenvalues)) / abs(min(eigenvalues)))
    
    return values

if __name__ == '__main__':
    mesh = io.read_triangle_mesh('./output/preprocess/m100.off')
    mesh = mesh if not mesh.is_empty() else io.read_triangle_mesh('./data_out/m100.off') # NOTE: My data_out looks like this ~Simon
    A3 = angle_between_randoms(mesh)
    norm_A3 = normalize_features(A3)
    hist_A3 = visualize_histogram(normalize_features(norm_A3), "3_random_angles", "./output/hist/")
    D1 = distance_barycenter_to_random(mesh)
    norm_D1 = normalize_features(D1)
    hist_D1 = visualize_histogram(norm_D1, "barycenter_to_random", "./output/hist/")
    D2 = distance_random_to_random(mesh)
    norm_D2 = normalize_features(D2)
    hist_D2 = visualize_histogram(norm_D2, "random_to_random", "./output/hist/")

    # #TODO: Implement D3 and D4
