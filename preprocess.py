'''Module for preprocessing the off and ply files'''
from logging import getLogger
from os import mkdir
from os.path import basename, exists, isfile
from posixpath import dirname
from re import search
from typing import Dict, Tuple

import trimesh as tm
import pymeshfix
import numpy as np
import pandas as pd

from extraction import extract_all_features
from util import locate_mesh_files, calculate_mesh_center, SIZE_PARAM

log = getLogger('preprocess')
SIZE_MAX = SIZE_PARAM + int(SIZE_PARAM * 0.2)
SIZE_MIN = SIZE_PARAM - int(SIZE_PARAM * 0.2)

convert_to_trimesh = lambda x: tm.Trimesh(np.asarray(x.vertices), np.asarray(x.triangles))

def sub_sample(current_mesh: tm.Trimesh) -> tm.Trimesh:
    '''Subsamples a mesh to decrease the number of vertices using quadratic error measure decimation

    Args:
        current_mesh (tm.Trimesh): The mesh to decimate

    Returns:
        tm.Trimesh: The decimated mesh
    '''
    current_mesh = current_mesh.as_open3d.simplify_quadric_decimation(target_number_of_triangles=SIZE_PARAM)
    current_mesh = convert_to_trimesh(current_mesh)

    return current_mesh

def super_sample(current_mesh: tm.Trimesh) -> tm.Trimesh:
    '''Supersamples a mesh to increase the number of vertices using midpoint division

    Args:
        current_mesh (tm.Trimesh): The mesh to supersample

    Returns:
        tm.Trimesh: The supersampled mesh
    '''
    current_mesh = current_mesh.as_open3d.subdivide_midpoint()
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
    
    current_mesh = convert_to_trimesh(current_mesh)
    
    return current_mesh

def find_aabb_points (current_mesh: tm.Trimesh) -> Tuple[np.ndarray, np.ndarray]:
    '''Finds the AABB boundry points

    Args:
        current_mesh (tm.Trimesh): The mesh to check

    Returns:
        Tuple[np.ndarray, np.ndarray]: The minimum and maximum points as a tuple
    '''
    bounds = current_mesh.bounds
    aabb_min = bounds[0]
    aabb_max = bounds[1]

    return (aabb_min, aabb_max)

def acceptable_size(face_count: int) -> bool:
    '''Checks whether a shape is in acceptable size range from it's face count

    Args:
        face_count (int): The face count to check

    Returns:
        bool: Whether the shape is in acceptable range
    '''
    return (face_count <= SIZE_PARAM and face_count > SIZE_MIN) or (face_count >= SIZE_PARAM and face_count < SIZE_MAX)

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

    for file in files:
        log.debug('Preprocessing file: %s', file)
        current_mesh = tm.load(file)  

        if current_mesh.is_empty:
            log.error('Mesh at %s could not be read.', file)
            continue

        # Step 1: Get Information
        label = basename(dirname(file))
        face_count = current_mesh.faces.shape[0]
        vertex_count = current_mesh.vertices.shape[0]

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

        #verify closed mesh
        current_mesh = make_watertight(current_mesh)

        # Step 3: Normalize
        current_mesh = normalize_mesh(current_mesh)

        # Step 4: Get aabb points
        # DATA ENTRY: Min and Max aabb points
        aabb_min, aabb_max = find_aabb_points(current_mesh)

        # Step 5: Extract Features and save them inside the database.
        features = extract_all_features(current_mesh)

        # DATA ENTRY: Face and Vertex Count 
        face_count = current_mesh.faces.shape[0]
        vertex_count = current_mesh.vertices.shape[0]

        # Create new
        mesh_name  = basename(file)[:-4]
        extended_output_path  = './' + output_path + '/' + mesh_name + '/'

        if not exists (extended_output_path):
            mkdir(extended_output_path)

        current_mesh_path = extended_output_path + basename(file)
        tm.exchange.export.export_mesh(current_mesh, current_mesh_path,'off')

        entry = {
            'aabb_max': aabb_max,
            'aabb_min': aabb_min,
            'label': label,
            'face_count': face_count,
            'vertex_count': vertex_count,
            'path': current_mesh_path
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

def single_preprocess(file_path: str) -> Dict[str, str]:
    '''Preprocesses a single file and returns the database entry

    Args:
        file_path (str): The path to the file to preprocess
        classification_path (str, optional): The file to the classification file. Defaults to None.

    Returns:
        Dict[str, str]: A map from feature and datapoint names to their values
    '''
    # Load in the mesh
    if not isfile(file_path):
        log.critical('Input path "%s" not valid', file_path)
        return None

    mesh = tm.load(file_path)
    
    if mesh.is_empty:
        log.critical('Failed to read triangle mesh from file "%s"', file_path)
        return None

    # Create a dictionary to store features
    entry = {'path': file_path}
    labels = None

    # Assign label if possible
    if labels is not None:
        fnm = search(r'\d+', basename(file_path))

        if fnm is not None and fnm[0] in labels:
            entry['label'] = labels[fnm[0]]
            log.debug('Labeled mesh %s as %s', basename(file_path), entry['label'])
    
    # Fix and normalize the mesh, and get the features
    mesh = make_watertight(mesh) 
    mesh = normalize_mesh(mesh)
    aabb_min, aabb_max = find_aabb_points(mesh)
    features = extract_all_features(mesh, 200)

    # Create the final entry
    entry.update({
        'aabb_max': aabb_max,
        'aabb_min': aabb_min,
        'face_count': mesh.faces.shape[0],
        'vertex_count': mesh.vertices.shape[0]
    })
    entry.update(features)

    # Return the created entry
    return entry

def flip_test(mesh: tm.Trimesh) -> tm.Trimesh:
    '''Function that mirrors the mesh if necessary on the x,y or z axis.

    Args:
        mesh (tm.Trimesh): The mesh to mirror

    Returns:
        tm.Trimesh: The mirrored mesh
    '''
    f_sign = [0,0,0]

    vertices =  mesh.vertices

    for face in mesh.faces:

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

    mesh.vertices = np_vertices

    return mesh

def pose_alignment(mesh: tm.Trimesh) -> tm.Trimesh:
    '''Function that aligns the mesh to the xyz axis.

    Args:
        mesh (tm.Trimesh): The mesh to align

    Returns:
        tm.Trimesh: The axis-aligned mesh
    '''

    pca = mesh.principal_inertia_vectors
    z_axis = np.cross(pca[0], pca[1])
    centroid = calculate_mesh_center(mesh) # Maybe not needed anymore since we are already at 0

    vertices = []

    for vertex in np.asarray(mesh.vertices):

        coords =  vertex - centroid

        x_coord = np.dot(coords, pca[0])
        y_coord = np.dot(coords, pca[1])
        z_coord = np.dot(coords, z_axis)

        projected_vertex = np.asarray([x_coord, y_coord, z_coord])
        vertices.append(projected_vertex)
    
    np_vertices = np.stack(vertices)

    mesh.vertices = np_vertices

    return mesh

def scale_mesh(mesh: tm.Trimesh) -> tm.Trimesh:
    '''Function that scales the mesh to fit into an unit cube.

    Args:
        mesh (tm.Trimesh): The mesh to scale

    Returns:
        tm.Trimesh: The scaled mesh
    '''
    
    min_bound, max_bound = find_aabb_points(mesh)

    diff = abs(max_bound - min_bound)
    max_dim = None

    for i, value in enumerate(diff):
        if max_dim is None or value > diff[max_dim]:
            max_dim = i

    scale_factor = 1 / diff[max_dim]
    
    transformation = np.zeros([4,4])
    transformation[3,3] = 1
    for i in range(0,3):
        transformation[i,i] = scale_factor

    return mesh.apply_transform(transformation)

def make_watertight(mesh: tm.Trimesh) -> tm.Trimesh:
    '''Attempts to make a mesh watertight

    Args:
        mesh (tm.Trimesh): The mesh to make watertight

    Returns:
        tm.Trimesh: The closed mesh
    '''
    if not mesh.is_watertight:
        vclean, fclean = pymeshfix.clean_from_arrays(mesh.vertices, mesh.faces)
        mesh = tm.Trimesh(vclean, fclean)
        log.error(f'Non-watertightness identified and fix attempted, success: {mesh.is_watertight}')

    return mesh

def translate(mesh: tm.Trimesh) -> tm.Trimesh:
    translation = calculate_mesh_center(mesh)

    transformation = np.zeros([4,4])

    for i in range(0,4):
        transformation[i,i] = 1

    for i in range(0,3):
        transformation[i,3] = -translation[i]

    mesh.apply_transform(transformation)

    return mesh

def normalize_mesh(mesh: tm.Trimesh) -> tm.Trimesh:
    '''Function that calls every necessary normalization step.

    Args:
        mesh (tm.Trimesh): The mesh to normalize

    Returns:
        tm.Trimesh: The normalized mesh
    '''

    mesh = mesh.apply_transform(mesh.principal_inertia_transform) 

    mesh.vertices = mesh.vertices - mesh.center_mass

    # STEP 3: Flip test
    mesh = flip_test(mesh)

    # STEP 4: Scale
    mesh = scale_mesh(mesh)

    return mesh
