from typing import Tuple

import numpy as np
import trimesh as tm
from trimesh.poses import compute_stable_poses

def principle_component_analysis(vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''Performs principle component analysis returning the three eigenvectors and the eigenvalues

    Args:
        vertices (np.ndarray): The mesh vertices to analyze

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the eigenvalues and eigenvectors respectively in a (1, M) and (M, M) numpy array
    '''
    # Get the eigenvectors and eigenvalues
    adjusted_vertices = vertices - np.mean(vertices)
    matrix = np.cov(adjusted_vertices.T)
    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    # Sort the eigenvectors and eigenvalues, normalize the eigenvectors to a length of 1
    sort = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[sort]
    eigenvectors = np.apply_along_axis(lambda vec: vec / np.sqrt(np.sum(vec ** 2)), 1, eigenvectors[sort])

    # Return the eigenvalues and eigenvectors
    return eigenvalues, eigenvectors

def alignment_fix(vertices: np.ndarray, eigenvectors: np.ndarray) -> np.ndarray:
    center = np.mean(vertices)
    projection = []

    for vertex in vertices:
        projection.append(np.asarray([
            np.dot(vertex - center, eigenvectors[0]),
            np.dot(vertex - center, eigenvectors[1]),
            np.dot(vertex - center, eigenvectors[2])
        ]))
    
    return np.stack(projection)

if __name__ == '__main__':
    mesh = tm.load('./princeton/1/m100/m100.off')
    mesh.vertices = mesh.vertices - mesh.center_mass
    #vals, vecs = principle_component_analysis(np.asarray(mesh.vertices))
    print(mesh.vertices.shape)
    cspt, prbs = compute_stable_poses(mesh)
    #mesh.vertices = alignment_fix(mesh.vertices, vecs)
    vecs = mesh.principal_inertia_vectors

    meshb = tm.Trimesh(mesh.vertices, mesh.faces)
    meshb.apply_transform(cspt[np.argmax(prbs),:,:])
    pc = tm.PointCloud(np.asarray([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            vecs[0],
            vecs[1],
            vecs[2]
        ]), colors=np.asarray([
            [0, 0, 0, 255],
            [255, 0, 0, 255],
            [0, 255, 0, 255],
            [0, 0, 255, 255],
            [128, 0, 0, 255],
            [0, 128, 0, 255],
            [0, 0, 128, 255]
        ]))
    scene = tm.Scene([mesh, pc])
    scene.show('gl')