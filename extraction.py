'''Module for extracting features from 3D meshes'''
from logging import getLogger
from math import pi, sqrt, isnan
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union
from util import calculate_mesh_center, BIN_COUNT, SAMPLE_SIZE

import numpy as np
import matplotlib.pyplot as plt
import trimesh as tm
import matplotlib.pyplot as plt

log = getLogger('features')

def angle_between_randoms(mesh: tm.Trimesh, samples: int = SAMPLE_SIZE) -> List[float]:
    '''Calculates angle between 3 random vertices.
    
    Args:
        mesh (tm.Trimesh): The mesh from which we select the points
        num_of_points (int): Number of points to calculate
    
    Returns:
        List[float]: The list of angles between 3 random vertices
    '''
    vertices = mesh.vertices
    entries = []
    sample_count = int((samples)**(1.0/3.0))
    entries = []

    for _ in range(0, sample_count):
        i_index = np.random.choice(vertices.shape[0], size=1, replace= False)
        v_i = vertices[i_index, :][0]

        for _ in range(0, sample_count):
            j_index = np.random.choice(vertices.shape[0], size=1, replace= False)
            v_j = vertices[j_index, :][0]
            if j_index == i_index:
                continue

            for k in range(0,sample_count):
                k_index = np.random.choice(vertices.shape[0], size=1, replace= False)
                v_k = vertices[k_index, :][0]
                
                if k_index == i_index or k_index == j_index:
                    continue
            
                vector_a_to_b = v_i - v_j
                vector_c_to_b = v_k - v_j

                cos_angle = np.dot(vector_a_to_b, vector_c_to_b) / (np.linalg.norm(vector_a_to_b) * np.linalg.norm(vector_c_to_b))
                angle = np.arccos(cos_angle)
                
                # This normalizes the angles to values between 0 - 1, facilitates histogram creation.
                f_angle = np.degrees(angle)/360
                
                if isnan(f_angle) is True:
                    continue

                entries.append(f_angle)    

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

def distance_barycenter_to_random(mesh: tm.Trimesh, samples: int = SAMPLE_SIZE) -> List[float]:
    '''Calculates distance from barycenter to random vertices
    
    Args:
        mesh (tm.Trimesh): The mesh from which we select the points
        samples (int): Number of points to calculate distance for
    
    Returns:
        List[float]: The distances between the barycenter of the mesh and random vertices
    '''
    barycenter = calculate_mesh_center(mesh)
    vertices = np.asarray(mesh.vertices)
    entries = []

    for _ in range(samples):
        random_indeces = np.random.choice(vertices.shape[0], size=1, replace= False)
        random_vertex = vertices[random_indeces, :]
        dist = distance_between_points(barycenter, random_vertex[0])
        entries.append(dist)
    
    return entries

def distance_random_to_random(mesh: tm.Trimesh, samples: int = SAMPLE_SIZE) -> List[float]:
    '''Calculates distance from a random vertex to another random vertex.
    
    Args:
        mesh (tm.Trimesh): The mesh from which we select the vertices
        samples (int): Number of point pairs to calculate distance for
    
    Returns:
        List[float]: The distance between 2 random vertices
    '''
    vertices = mesh.vertices

    sample_count = int((samples)**(1.0/2.0))

    entries = []

    for i in range(0, sample_count):
        i_index = np.random.choice(vertices.shape[0], size=1, replace= False)
        v_i = vertices[i_index, :][0]

        for j in range(0, sample_count):
            j_index = np.random.choice(vertices.shape[0], size=1, replace= False)
            v_j = vertices[j_index, :][0]
            if j_index == i_index:
                continue
            dist = distance_between_points(v_i, v_j)
            entries.append(dist)


    return entries

def create_histogram(data: List[float], bin_count: int = BIN_COUNT, normalized: bool = True) -> Tuple[List[float], List[int]]:
    '''Generates a histogram from a list of datapoints

    Args:
        data (List[float]): The list of datapoints
        bin_count (int, optional): The number of bins to divide the data over. Defaults to BIN_COUNT.
        normalized (bool, optional): Whether to normalize the data. Defaults to True.

    Returns:
        Tuple[List[int],List[float]]: A tuple respectively containing the counts per bin and the bin boundries both as a list
    '''
    counts, bins = np.histogram(data, bins = bin_count)
    if normalized:
        counts = normalize_histogram(counts)

    return counts, bins


def normalize_histogram(entries: List[float]) -> List[float]:
    '''Normalizes a histogram

    Args:
        entries (List[float]): The unnormalized histogram

    Returns:
        List[float]: The histogram, normalized to a range 0-1
    '''
    total = sum(entries)
    entries = [entry/total for entry in entries]

    return entries

def visualize_histogram(counts: np.array, bins: np.array, title: str, output_path: str) -> None:
    '''Generates a visual representation of a histogram and prints it to a PNG file

    Args:
        counts (np.array): The array containing the counts of the histogram
        bins (np.array): An array containing the boundries of the histogram
        title (str): The title for the graph
        output_path (str): The output path for the image
    '''
    plt.close() # Close any previously opened plots
    plt.hist(bins[:-1], bins, weights=counts) # Create the histogram
    # Set the title, y axis label, and the x axis label
    plt.title(title) 
    plt.ylabel("Frequency")
    plt.xlabel("Range")
    # Save and close the plot
    plt.savefig(output_path)
    plt.close()

def volume_of_random_vertices(mesh: tm.Trimesh, samples: int = SAMPLE_SIZE) -> List[float]:
    '''Calculates the volume of four randomly chosen vertices, multiple times

    Args:
        mesh (tm.Trimesh): The mesh to sample
        samples (int, optional): The number of samples to take. Defaults to SAMPLE_SIZE.

    Returns:
        List[float]: A list of float results
    '''
    vertices = mesh.vertices
    sample_count = int((samples)**(1.0/4.0))
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
                    
                    tetra_vertices = np.asarray([v_i,v_j,v_k,v_l])

                    tetra_faces =np.asarray([[0,3,1], [2,3,0], [3,2,1], [1,2,0]])

                    # Avoid doing calcs by hand :)
                    tetrahedron = tm.Trimesh(vertices = tetra_vertices, faces = tetra_faces)
                    
                    tetrahedron.fix_normals()

                    volume = tetrahedron.mass_properties['volume'] 
                    cr_volume = volume **(1./3)
                    if not isnan(cr_volume):
                        entries.append(cr_volume)

    return entries

def area_of_random_vertices(mesh: tm.Trimesh, samples: int = SAMPLE_SIZE) -> List[float]:
    """Calculates area of a triangle made from 3 random vertices.
    
    Args:
        mesh (tm.Trimesh): The mesh from which we select the vertices
        samples (int): Number of samples to evaluate the area for.
    
    Returns:
        List[float]: The area created by a triangle of three random vertices.
    """

    vertices = mesh.vertices
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
                
                # find vectors i_j, i_k
                i_j = [v_i[0] - v_j[0], v_i[1] - v_j[1], v_i[2] - v_j[2]]
                i_k = [v_i[0] - v_k[0], v_i[1] - v_k[1], v_i[2] - v_k[2]]

                # get their cross product
                vectors = np.cross(i_j,i_k)

                # calculate the magnitude of cross product
                area = abs(np.sqrt(vectors.dot(vectors)))/2

                sqrt_area = sqrt(area)

                entries.append(sqrt_area)

    return entries


def simple_features(mesh: tm.Trimesh) -> Dict[str, float]:
    '''Extract some simple features from a 3D mesh

    Args:
        mesh (tm.Trimesh): The mesh to extract features from

    Returns:
        Dict[str, float]: A list of features, in order: surface area, compactness, AABB volume, diameter, and eccentricity
    '''
    values = {}

    # Get the surface area
    surface_area = mesh.area
    values['surface_area'] = surface_area

    # Get compactness
    if mesh.is_watertight:
        mesh.fix_normals()
        values['compactness'] = (surface_area ** 3) / (36 * pi * (mesh.mass_properties['volume'] ** 2))
    else:
        values['compactness'] = None
        log.error("Cannot find compactness of a non-watertight mesh")
    # Get AABB volume
    values['aabb_volume'] = mesh.as_open3d.get_axis_aligned_bounding_box().volume()

    # Get diameter
    # This is using a bruteforce method, but relying mostly on numpy's C code, so it takes ~0.5 seconds for 2000 vertices
    vertices = np.asarray(mesh.vertices)
    vertex_count = vertices.shape[0]
    squared_distances = np.zeros((vertex_count, vertex_count))

    for i in range(vertex_count):
        squared_distances[i,:] = np.sum(np.square(vertices[i,:] - vertices), axis=1)

    values['diameter'] = np.sqrt(np.max(squared_distances))

    # Get eccentricity
    eigenvalues = mesh.principal_inertia_components
    values['eccentricity'] = abs(max(eigenvalues)) / abs(min(eigenvalues))
    
    return values

def distribution_features(mesh: tm.Trimesh, samples: int = SAMPLE_SIZE) -> Dict[str, np.array]:
    '''Gets the values of the distribution features for a shape

    Args:
        mesh (tm.Trimesh): The mesh to inspect
        samples (int, optional): The number of samples to take for each distribution feature. Defaults to SAMPLE_SIZE.

    Returns:
        Dict[str, np.array]: A dictionary mapping the feature names to their histograms
    '''
    dist = {}

    A3 = angle_between_randoms(mesh, samples)
    A3_counts, _ = create_histogram(data = A3, bin_count = BIN_COUNT)
    dist['A3'] = A3_counts
    
    D1 = distance_barycenter_to_random(mesh, samples)
    D1_counts, _ = create_histogram(data = D1, bin_count = BIN_COUNT)
    dist['D1'] = D1_counts

    D2 = distance_random_to_random(mesh, samples)
    D2_counts, _ = create_histogram(data = D2, bin_count = BIN_COUNT)
    dist['D2'] = D2_counts

    D3 = area_of_random_vertices(mesh, samples)
    D3_counts, _ = create_histogram(data = D3, bin_count = BIN_COUNT)
    dist['D3'] = D3_counts
    
    D4 = volume_of_random_vertices(mesh, samples)
    D4_counts, _ = create_histogram(data = D4, bin_count = BIN_COUNT)
    dist['D4'] = D4_counts

    return dist

def extract_all_features(mesh: tm.Trimesh, samples: int = SAMPLE_SIZE) -> Dict[str, Union[float, np.ndarray]]:
    '''Extracts simple and shape property features from a mesh

    Args:
        mesh (geometry.TriangleMesh): The mesh to extract features from
        samples (int, optional): The number of samples to take in the distribution features. Defaults to SAMPLE_SIZE.

    Returns:
        Dict[str, Union[float, np.ndarray]]: A dictionary mapping feature names to their values for this mesh
    '''
    # Get single features
    features = simple_features(mesh)

    # Add distributions to features
    distributions = distribution_features(mesh, samples)

    features.update(distributions)

    return features
    