'''Module for preprocessing the off and ply files'''
from logging import getLogger
from os import mkdir
from os.path import basename, exists, isfile
from re import search, split
from typing import Dict
from open3d import geometry, io, utility
from extraction import extract_all_features
from util import compute_pca, locate_mesh_files

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
    
    files = locate_mesh_files(input_path)
    
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

        # Step 4: Normalize
        current_mesh = normalize_mesh(current_mesh)

        # Step 3: Get aabb points
        # DATA ENTRY: Min and Max aabb points
        aabb_min, aabb_max = find_aabb_points(current_mesh)

        # Step 5: Extract Features and save them inside the database.
        features = extract_all_features(current_mesh)

        # DATA ENTRY: Face and Vertex Count 
        face_count = len(current_mesh.triangles)
        vertex_count = len(current_mesh.vertices)

        # DATA ENTRY: Mesh path
        current_mesh_path = output_path + '/' + basename(file)
        io.write_triangle_mesh(current_mesh_path, current_mesh)

        entry = {
            'aabb_max': aabb_max,
            'aabb_min': aabb_min,
            'label': label,
            'face_count': face_count,
            'path': current_mesh_path,
            'vertex_count': vertex_count
        } 

        if not entry:
            log.error('Shape at %s could not be converted to a dictionary, excluded from database.')
            continue
        
        entry.update(features)

        preprocessed_files.append(entry)

    # This will be the database we can load from.
    dataframe = pd.DataFrame(preprocessed_files)
    dataframe.to_csv(output_path + '/database.csv', sep=',', index=False)
    log.info('File data saved to "%s"', output_path + '/database.csv')

def single_preprocess(file_path: str, classification_path: str = None) -> Dict[str, str]:
    '''Preprocesses a single file and returns the database entry

    Args:
        file_path (str): The path to the file to preprocess
        classification_path (str, optional): The file to the classification file. Defaults to None.

    Returns:
        Dict[str, str]: A map from feature and datapoint names to their values
    '''
    if not exists(file_path) or not isfile(file_path):
        log.critical('Input path "%s" not valid', file_path)
        return None

    mesh = io.read_triangle_mesh(file_path)
    
    if mesh.is_empty():
        log.critical('Failed to read triangle mesh from file "%s"', file_path)
        return None

    entry = {'path': file_path}
    labels = None

    if classification_path is not None and isfile(classification_path):
        labels = get_labels(classification_path)

    if labels is not None:
        fnm = search(r'\d+', basename(file_path))

        if fnm is not None and fnm[0] in labels:
            entry['label'] = labels[fnm[0]]
            log.debug('Labeled mesh %s as %s', basename(file_path), entry['label'])
    
    mesh = normalize_mesh(mesh)
    aabb_min, aabb_max = find_aabb_points(mesh)
    features = extract_all_features(mesh)

    entry.update({
        'aabb_max': aabb_max,
        'aabb_min': aabb_min,
        'face_count': len(mesh.triangles),
        'vertex_count': len(mesh.vertices)
    })
    entry.update(features)

    return entry

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

def flip_test(mesh: geometry.TriangleMesh) -> geometry.TriangleMesh:
    '''Function that mirrors the mesh if necessary on the x,y or z axis.

    Args:
        mesh (geometry.TriangleMesh): The mesh to mirror

    Returns:
        geometry.TriangleMesh: The mirrored mesh
    '''
    f_sign = [0,0,0]

    vertices =  np.asarray(mesh.vertices)

    for face in np.asarray(mesh.triangles):

        vertex_total = np.zeros(3,dtype=float)

        for vertex in face:
            vertex_total+=vertices[vertex]

        center_of_face = vertex_total/3

        for i,coord in enumerate(center_of_face):
            if coord >= 0:
                f_sign[i] += 1
            else:
                f_sign[i] -= 1

    f_sign = [1 if r >= 0 else -1 for i,r in enumerate(f_sign) ]

    updated_vertices = []

    for vertex in vertices:
        mirror = vertex * f_sign
        updated_vertices.append(mirror)
    
    np_vertices = np.stack(updated_vertices)

    mesh.vertices = utility.Vector3dVector(np_vertices)

    return mesh

def pose_alignment(mesh: geometry.TriangleMesh) -> geometry.TriangleMesh:
    '''Function that aligns the mesh to the xyz axis.

    Args:
        mesh (geometry.TriangleMesh): The mesh to align

    Returns:
        geometry.TriangleMesh: The axis-aligned mesh
    '''
    x_axis, y_axis, z_axis, _ = compute_pca(mesh)
    centroid = mesh.get_center() # Maybe not needed anymore since we are already at 0

    vertices = []
    # R = np.stack([x_axis, y_axis ,z_axis])
    # # print(R.shape)
    # mesh = mesh.rotate(R,center = centroid)

    # return mesh
    for vertex in np.asarray(mesh.vertices):

        coords =  vertex - centroid

        x_coord = np.dot(coords, x_axis)
        y_coord = np.dot(coords, y_axis)
        z_coord = np.dot(coords, z_axis)

        projected_vertex = np.asarray([x_coord, y_coord, z_coord])
        vertices.append(projected_vertex)
    
    np_vertices = np.stack(vertices)

    mesh.vertices = utility.Vector3dVector(np_vertices)

    return mesh

def scale_mesh(mesh: geometry.TriangleMesh) -> geometry.TriangleMesh:
    '''Function that scales the mesh to fit into an unit cube.

    Args:
        mesh (geometry.TriangleMesh): The mesh to scale

    Returns:
        geometry.TriangleMesh: The scaled mesh
    '''
    aabb = mesh.get_axis_aligned_bounding_box()
    max_bound = aabb.get_max_bound()
    mesh_center = mesh.get_center()
    min_bound = aabb.get_min_bound()

    diff = abs(max_bound - min_bound)
    max_dim = None

    for i, value in enumerate(diff):
        if max_dim is None or value > diff[max_dim]:
            max_dim = i

    scale_factor = 1 / diff[max_dim]

    return mesh.scale(scale_factor, mesh_center)

# TODO: Add mesh repair to mesh normalization
def normalize_mesh(mesh: geometry.TriangleMesh) -> geometry.TriangleMesh:
    '''Function that calls every necessary normalization step.

    Args:
        mesh (geometry.TriangleMesh): The mesh to normalize

    Returns:
        geometry.TriangleMesh: The normalized mesh
    '''
    mesh_center = mesh.get_center()

    # STEP 1: Translate
    mesh = mesh.translate(-mesh_center)

    # STEP 2: Align
    mesh = pose_alignment(mesh)

    # STEP 3: Flip test
    mesh = flip_test(mesh)

    # STEP 4: Scale
    mesh = scale_mesh(mesh)

    return mesh
