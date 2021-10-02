'''Module for preprocessing the off and ply files'''
from logging import getLogger
from os import getcwd, listdir, mkdir
from os.path import basename, exists, isfile, isdir
from re import match, search, split
from typing import Dict
from open3d import geometry, io, utility
import numpy as np
import pandas as pd

log = getLogger('preprocess')
SIZE_PARAM = 3500 # Check which value we want to use.
SIZE_MAX = SIZE_PARAM + int(SIZE_PARAM * 0.2)
SIZE_MIN = SIZE_PARAM - int(SIZE_PARAM * 0.2)


def sub_sample(current_mesh: geometry.TriangleMesh) -> geometry.TriangleMesh:
    current_mesh = current_mesh.simplify_quadric_decimation(target_number_of_triangles=SIZE_PARAM)
    return current_mesh

def super_sample(current_mesh: geometry.TriangleMesh) -> geometry.TriangleMesh:
    current_mesh = current_mesh.subdivide_midpoint()

    face_count = len(current_mesh.triangles)

    not_ready = True

    while not_ready:
        if face_count < SIZE_MIN:
                current_mesh = current_mesh.subdivide_midpoint()
        elif face_count > SIZE_MAX:
                current_mesh = current_mesh.simplify_quadric_decimation(target_number_of_triangles=SIZE_PARAM)
        else:
            not_ready = False

        face_count = len(current_mesh.triangles)
    
    return current_mesh

def find_aabb_points (current_mesh: geometry.TriangleMesh) -> tuple():

    aabb = current_mesh.get_axis_aligned_bounding_box()

    aabb_min = aabb.get_min_bound()
    aabb_max = aabb.get_max_bound()

    return (aabb_min, aabb_max)

def acceptable_size(face_count: int) -> bool:

    if face_count <= SIZE_PARAM and face_count > SIZE_MIN:
        return True
    
    if face_count >= SIZE_PARAM and face_count < SIZE_MAX:
        return True

    return False

def preprocess(input_path: str, output_path: str, classification_path: str) -> None:
    '''Function that preprocesses files from input to output

    Args:
        input_path (str): The input file/folder
        output_path (str): The output folder
        classifaction_path (str): The path to the classification files
    '''
    # Check if the input is valid
    if not exists(input_path):
        log.critical('Input path "%s" does not exist', input_path)
        return
    
    # Check if the output is valid, otherwise try to create it
    if isfile(output_path):
        log.critical('Output path "%s" points to a file, but should point to a folder', output_path)
        return

    if not exists(output_path):
        log.info('Output path "%s" does not exists, creating it for you', output_path)
        mkdir(output_path)
    
    # Check if the classification_path is valid, if it's provided
    if classification_path is not None and not exists(classification_path):
        log.warning('The provided path to the classification file "%s" is not valid', classification_path)
        classification_path = None

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
    
    log.debug('Found %d files to preprocess', len(files))

    preprocessed_files = []

    # Get label data
    labels = get_labels(classification_path) if classification_path is not None else None

    for file in files:
        log.debug('Preprocessing file: %s', file)
        current_mesh = io.read_triangle_mesh(file)  

        if current_mesh.is_empty():
            log.error('Mesh at %s could not be read.', file)
            continue

        # Step 1: Get Information
        label = None
        face_count = len(current_mesh.triangles)
        vertex_count = len(current_mesh.vertices)
        log.debug('Face: (%d) and Vertex (%d)', face_count, vertex_count)

        # Find category if relevant
        if labels is not None:
            fnm = search(r'\d+', basename(file))
        
        # DATA ENTRY: Label 
            if fnm is not None and fnm[0] in labels:
                label = labels[fnm[0]]
                log.debug('Labeled mesh %s as %s', basename(file), label)

        # Step 2: Supersample/Subsample
        if not acceptable_size(face_count):

            if face_count > SIZE_MAX:
                current_mesh = sub_sample(current_mesh)
                log.debug("Decimated shape %s, previously (%d) faces and (%d) vertices, currently (%d) faces and (%d) vertices", 
                            file, face_count, vertex_count, len(current_mesh.triangles), len(current_mesh.vertices))
                
            elif face_count < SIZE_MIN:
                
                current_mesh = super_sample(current_mesh)
                log.debug("Supersampled shape %s, previously (%d) faces and (%d) vertices, currently (%d) faces and (%d) vertices", 
                           file, face_count, vertex_count, len(current_mesh.triangles), len(current_mesh.vertices))

        # Step 3: Get aabb points
        # DATA ENTRY: Min and Max aabb points
        aabb_min, aabb_max = find_aabb_points(current_mesh)

        # TODO: Step 3.1, rotate mesh

        # Step 4: Normalize
        current_mesh = normalize_mesh(current_mesh)

        # DATA ENTRY :Face and Vertex Count 
        face_count = len(current_mesh.triangles)
        vertex_count = len(current_mesh.vertices)

        # DATA ENTRY: Mesh path
        current_mesh_path = output_path + '/' + basename(file)
        io.write_triangle_mesh(current_mesh_path, current_mesh)

        entry = {'path': current_mesh_path, 'label': label, 
                    'face_count': face_count, 'vertex_count': vertex_count} 
        if not entry:
            log.error('Shape at %s could not be converted to a dictionary, excluded from database.')
            continue

        preprocessed_files.append(entry)


    # This will be the database we can load from.
    dataframe = pd.DataFrame(preprocessed_files)
    dataframe.to_csv(output_path + '/df.csv')

    return

def get_labels(path: str) -> Dict[str, str]:
    '''Retrieves the labels from a relevant cla file

    Args:
        path (str): The path to the cla file

    Returns:
        Dict[str, str]: Mapping from filename to label
    '''
    mapping = dict()
    
    def getline(f):
        l = f.readline()
        return split(r'\s+', l.lower().strip()) if l != '' else None

    with open(path, 'r') as f:
        line = getline(f)

        # Check first line
        if line[0] != 'psb':
            log.warning('Can not acquire labels from "%s", does not follow princeton specification')
            return mapping

        # Get category and file count
        getline(f) # Discard second line
        line = getline(f)
        category = None

        while line is not None:
            if search(r'^([a-zA-Z]+|\-1)', line[0]) is not None:
                category = line[0]

            if category is None:
                line = getline(f)
                continue

            if search(r'\d+', line[0]) is not None:
                mapping[line[0]] = category

            line = getline(f)

    log.debug('Retrieved %d labels from "%s"', len(mapping), path)
    return mapping

def compute_PCA(mesh:geometry.TriangleMesh):
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

    return eigenvalues, eigenvectors


def compute_OBB(mesh: geometry.TriangleMesh) -> np.array:
    pass

def sort_eigen_vectors(eigen_values: np.array, eigen_vectors: np.array):

    eigen_values, eigen_vectors = zip(*sorted(zip(eigen_values,eigen_vectors), reverse=True))
    eigen_vectors = np.asarray(eigen_vectors)

    return eigen_values, eigen_vectors

def pose_normalization(mesh: geometry.TriangleMesh) -> geometry.TriangleMesh:

    log.error('Pose normalization is not complete.')

    eigen_values, eigen_vectors= compute_PCA(mesh)
    eigen_values, eigen_vectors = sort_eigen_vectors(eigen_values, eigen_vectors)

    x_axis =  eigen_vectors[0]
    y_axis = eigen_vectors[1]
    z_axis = eigen_vectors[0] * eigen_vectors [1]

    centroid = mesh.get_center()

    for vertex in mesh.vertices:
        x_coord = (vertex[0] - centroid) * x_axis
        y_coord = (vertex[1] - centroid) * y_axis
        z_coord = (vertex[2] - centroid) * z_axis

        projected_vertex = np.asarray([x_coord, y_coord, z_coord])
        pass
    return mesh


def normalize_mesh(mesh: geometry.TriangleMesh) -> geometry.TriangleMesh:
    '''Normalize the mesh to be scaled and translated to a unit cube around the origin

    Args:
        mesh (geometry.TriangleMesh): The mesh to normalize

    Returns:
        geometry.TriangleMesh: The normalized mesh
    '''
    aabb = mesh.get_axis_aligned_bounding_box()
    max_bound = aabb.get_max_bound()
    mesh_center = mesh.get_center() #We might need to calculate the centroid, not the center.
    min_bound = aabb.get_min_bound()

    # STEP 1: TRANSLATION, First translate object to center of the world

    mesh = mesh.translate(-mesh_center)

    mesh = pose_normalization(mesh)

        #Update points by aligning with the coordinate frame.
            # For this we do 
            # x = (px_i - c) dotproduct eigenvector1
            # y = (py_i - c) dotproduct eigenvector2
            # z = (pz_i - c) dotproduct (eigenvector1 x eigenvector2)  to get the min and max.
        # Now with the OBB 

    # STEP 3: FLIP test,

    # STEP 4: Scale, use OBB max and min to calculate scale factor.
    diff = abs(max_bound - min_bound)
    max_dim = None

    for i, value in enumerate(diff):
        #print(i)
        if max_dim is None or value > diff[max_dim]:
            max_dim = i

    scale_factor = 1 / diff[max_dim]

    return mesh.scale(scale_factor, mesh_center).translate(-mesh_center)


if __name__ == '__main__':

    mesh = io.read_triangle_mesh('./plane.ply')

    normalize_mesh(mesh)