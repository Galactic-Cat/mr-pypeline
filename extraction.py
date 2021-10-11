'''Module for extracting features from 3D meshes'''
from logging import getLogger
from math import floor, pi, sqrt
from typing import List

import numpy as np
from open3d import geometry, io, utility

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
        print(vertices.shape[0])
        print(vertices)
        random_indeces = np.random.choice(vertices.shape[0], size=3, replace= False)
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
    #vertex_count = vertices.shape[0] - (vertices.shape[0] % 2)
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
    plt.ylabel("Frequency")
    #plt.xlabel(x_label)
    plt.savefig(output_path)
    plt.close()

    return

def volume_of_random_vertices(mesh: geometry.TriangleMesh, samples: int = SAMPLE_SIZE):
    
    vertices = np.asarray(mesh.vertices)
    sample_count = int((samples)**(1.0/3.0))

    entries = []

    for i in range(0, sample_count):
        i_index = np.random.choice(vertices.shape[0], size=1, replace= False)
        v_i = vertices[i_index, :][0]

        for j in range(0, sample_count):
            j_index = np.random.choice(vertices.shape[0], size=1, replace= False)
            v_j = vertices[j_index, :][0]
            if j_index == i_index:
                continue

            for k in range(0,sample_count):
                k_index = np.random.choice(vertices.shape[0], size=1, replace= False)
                v_k = vertices[k_index, :][0]
                if k_index == i_index or k_index == j_index:
                    continue
                
                for l in range (0, sample_count):
                    l_index = np.random.choice(vertices.shape[0], size=1, replace= False)
                    v_l = vertices[l_index, :][0]
                    if l_index == i_index or l_index == j_index or l_index == k_index:
                        continue
                    
                    tetra_vertices = utility.Vector3dVector(np.asarray([v_i,v_j,v_k,v_l]))

                    tetra_faces = utility.Vector3iVector(np.asarray([[0,3,1], [2,3,0], [3,2,1], [1,2,0]]))

                    # Avoid doing calcs by hand :)
                    tetrahedron = geometry.TriangleMesh(tetra_vertices, tetra_faces)

                    volume = tetrahedron.get_volume()

                    entries.append(volume)

    return entries

def area_of_random_vertices(mesh: geometry.TriangleMesh, samples: int = SAMPLE_SIZE) -> List[float]:
    """Calculates area of a triangle made from 3 random vertices.
    
    Args:
        mesh (geometry.TriangleMesh): The mesh from which we select the vertices
        samples (int): Number of samples to evaluate the area for.
    
    Returns:
        List[float]: The area created by a triangle of three random vertices.
    """

    vertices = np.asarray(mesh.vertices)
    sample_count = int((samples*100)**(1.0/3.0))

    entries = []

    for i in range(0, sample_count):
        i_index = np.random.choice(vertices.shape[0], size=1, replace= False)
        v_i = vertices[i_index, :][0]

        for j in range(0, sample_count):
            j_index = np.random.choice(vertices.shape[0], size=1, replace= False)
            v_j = vertices[j_index, :][0]
            if j_index == i_index:
                continue

            for k in range(0,sample_count):
                k_index = np.random.choice(vertices.shape[0], size=1, replace= False)
                v_k = vertices[k_index, :][0]
                if k_index == i_index or k_index == j_index:
                    continue
                
                # find vectors i_j, i_k
                i_j = [v_i[0] - v_j[0], v_i[1] - v_j[1], v_i[2] - v_j[2]]
                i_k = [v_i[0] - v_k[0], v_i[1] - v_k[1], v_i[2] - v_k[2]]

                # get their cross product
                vectors = np.cross(i_j,i_k)

                # calculate the magnitude of cross product
                magnitude = abs(np.sqrt(vectors.dot(vectors)))

                area = magnitude/2

                entries.append(area)

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

# Leaving for debug purposes
# if __name__ == '__main__':
#     mesh = io.read_triangle_mesh('./output/preprocess/m393.off')
#     #mesh = mesh if not mesh.is_empty() else io.read_triangle_mesh('./data_out/m100.off') # NOTE: My data_out looks like this ~Simon

#     A3 = angle_between_randoms(mesh)
#     norm_A3 = normalize_features(A3)
#     hist_A3 = visualize_histogram(np.asarray(norm_A3), "Angles between 3 random vertices", "./output/hist_test/3_random_angles.png")
    
#     D1 = distance_barycenter_to_random(mesh)
#     norm_D1 = normalize_features(D1)
#     hist_D1 = visualize_histogram(np.asarray(norm_D1), "Distances between barycenter to random vertex", "./output/hist_test/barycenter_to_random.png")
    
#     D2 = distance_random_to_random(mesh)
#     norm_D2 = normalize_features(D2)
#     hist_D2 = visualize_histogram(np.asarray(norm_D2), "Distances between two random vertices", "./output/hist_test/random_to_random.png")

#     D3 = volume_of_random_vertices(mesh)
#     norm_D3 = normalize_features(D3)
#     hist_D3 = visualize_histogram(norm_D3, "Area of triangle from 3 random points", "./output/hist_test/area_3_random_vertices.png")
    
#     D4 = volume_of_random_vertices(mesh)
#     norm_D4 = normalize_features(D4)
#     hist_D4 = visualize_histogram(norm_D4, "Volume of tetahedron from 4 random points", "./output/hist_test/tetrahedron_area.png")
#     # simple_features(mesh)
