'''Module for preprocessing the off and ply files'''
from logging import getLogger
from open3d import geometry, io, utility
import numpy as np
import math

log = getLogger('feature_extraction')

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


def dist_b_to_r(mesh: geometry.TriangleMesh):
    '''Calculates distance from barycenter to a random vertex.
    
    Args:
        mesh (geometry.TriangleMesh): The mesh from which
        we select the points.
    
    Returns:
        float: The distance between the barycenter of the 
        mesh and a random point
    '''
    barycenter = mesh.get_center()

    vertices = np.asarray(mesh.vertices)

    number_of_vertices = vertices.shape[0]
    random_indeces = np.random.choice(number_of_vertices, size=1, replace= False)
    random_vertex = vertices[random_indeces, :]
    
    return dist_between_points(barycenter, random_vertex[0])

def dist_r_to_r(mesh: geometry.TriangleMesh):
    '''Calculates distance from a random vertex to a random vertex.
    
    Args:
        mesh (geometry.TriangleMesh): The mesh from which
        we select the points
    
    Returns:
        float: The distance between 2 random vertices
    '''

    vertices = np.asarray(mesh.vertices)

    number_of_vertices = vertices.shape[0]
    random_indeces = np.random.choice(number_of_vertices, size=2, replace= False)
    random_vertices = vertices[random_indeces, :]

    return dist_between_points(random_vertices[0], random_vertices[1])


def angle_between_3r(mesh: geometry.TriangleMesh):
    '''Calculates angle between 3 random vertices.
    
    Args:
        mesh (geometry.TriangleMesh): The mesh from which
        we select the points
    
    Returns:
        float: The angle between 3 random vertices
    '''

    vertices = np.asarray(mesh.vertices)

    number_of_vertices = vertices.shape[0]
    random_indeces = np.random.choice(number_of_vertices, size=3, replace= False)
    random_vertices = vertices[random_indeces, :]
    
    vector_b_to_a = random_vertices[0] - random_vertices[1]
    vector_b_to_c = random_vertices[2] - random_vertices[1]

    cos_angle = np.dot(vector_b_to_a, vector_b_to_c) / (np.linalg.norm(vector_b_to_a) * np.linalg.norm(vector_b_to_c))
    angle = np.arccos(cos_angle)

    return np.degrees(angle)

if __name__ == '__main__':
    mesh = io.read_triangle_mesh('./test_shapes/plane.ply')
