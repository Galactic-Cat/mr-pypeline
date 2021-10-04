'''Module for preprocessing the off and ply files'''
from logging import getLogger
from open3d import geometry, io, utility
import numpy as np
import math

log = getLogger('feature_extraction')

SAMPLE_SIZE = 250

def dist_between_points(vector_1: np.array, vector_2: np.array) -> float:
    '''Calculates distance between two points
    
    Args:
        vector_1 (np.array): Point one
        vector_2 (np.array): Point two
    
    Returns:
        float: The distance between the points.
    '''
    if vector_1 is None or vector_2 is None:
        log.error('Cannot find distance between null points.')
        return None

    total = 0

    for i,j in zip(vector_1, vector_2):
        total += (i-j)**2

    dist = math.sqrt(total)

    return dist


def dist_b_to_r(mesh: geometry.TriangleMesh,  num_of_points: int = SAMPLE_SIZE):
    '''Calculates distance from barycenter to a random vertex.
    
    Args:
        mesh (geometry.TriangleMesh): The mesh from which
        we select the points.
        num_of_points (int): Number of points to calculate
    
    Returns:
         list(float): The distance between the barycenter of the 
        mesh and a random point
    '''
    barycenter = mesh.get_center()

    vertices = np.asarray(mesh.vertices)
    number_of_vertices = vertices.shape[0]

    entries = []

    for point in range(num_of_points):
        random_indeces = np.random.choice(number_of_vertices, size=1, replace= False)
        random_vertex = vertices[random_indeces, :]
        dist = dist_between_points(barycenter, random_vertex[0])
        entries.append(dist)
    
    return entries

def dist_r_to_r(mesh: geometry.TriangleMesh, num_of_points: int = SAMPLE_SIZE):
    '''Calculates distance from a random vertex to a random vertex.
    
    Args:
        mesh (geometry.TriangleMesh): The mesh from which
        we select the points
        num_of_points (int): Number of points to calculate
    
    Returns:
         list(float): The distance between 2 random vertices
    '''

    vertices = np.asarray(mesh.vertices)

    number_of_vertices = vertices.shape[0]

    entries = []

    for point in range(num_of_points):
        random_indeces = np.random.choice(number_of_vertices, size=2, replace= False)
        random_vertices = vertices[random_indeces, :]
        dist = dist_between_points(random_vertices[0], random_vertices[1])
        entries.append(dist)

    return entries


def angle_between_3r(mesh: geometry.TriangleMesh, num_of_points: int = SAMPLE_SIZE):
    '''Calculates angle between 3 random vertices.
    
    Args:
        mesh (geometry.TriangleMesh): The mesh from which
        we select the points
        num_of_points (int): Number of points to calculate
    
    Returns:
        list(float): The list of angles between 3 random vertices
    '''

    vertices = np.asarray(mesh.vertices)

    number_of_vertices = vertices.shape[0]

    entries = []

    for point in range(num_of_points):
        random_indeces = np.random.choice(number_of_vertices, size=3, replace= False)
        random_vertices = vertices[random_indeces, :]
        
        vector_b_to_a = random_vertices[0] - random_vertices[1]
        vector_b_to_c = random_vertices[2] - random_vertices[1]

        cos_angle = np.dot(vector_b_to_a, vector_b_to_c) / (np.linalg.norm(vector_b_to_a) * np.linalg.norm(vector_b_to_c))
        angle = np.arccos(cos_angle)

        # This normalizes the angles to values between 0 - 1, facilitates histogram creation.
        f_angle = np.degrees(angle)/360

        entries.append(f_angle)

    return entries

def convert_entries_into_hist(entries: list, bin_count: int) -> np.array:
    '''Creates a histogram for a set of points.
    
    Args:
        entries(list): List of the points to convert to a distribution.
    
    Returns:
        np.array: contains the histogram of the points. 
    '''
    step_size = 1/bin_count

    hist = np.zeros(bin_count, dtype=int)

    for point in entries:
        #tells us where to put it  #TODO FIX: THIS PROBABLY BROKEN, floating point issues 
        index = math.floor(point/step_size)
        hist[index] +=1

    print(hist)
    return hist

if __name__ == '__main__':
    mesh = io.read_triangle_mesh('./data_out/database/m100.off')
    lister = dist_b_to_r(mesh,10)

    convert_entries_into_hist([0,.1,.2,.3,.4,.5,.6,.7,.8,.9], 10)
