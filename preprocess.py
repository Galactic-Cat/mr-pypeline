'''Module for preprocessing the off and ply files'''
from logging import getLogger
from os import listdir, mkdir
from os.path import basename, exists, isfile, isdir
from re import match, search, split
from shape import Shape
from typing import Dict

import pandas as pd

log = getLogger('preprocess')
SIZE_PARAM = 2000 # Check which value we want to use.
SIZE_MAX = SIZE_PARAM + int(SIZE_PARAM * 0.2)
SIZE_MIN = SIZE_PARAM - int(SIZE_PARAM * 0.2)

def acceptable_size(shape: Shape) -> bool:

    if shape.face_count <= SIZE_PARAM and shape.face_count > SIZE_MIN:
        return True
    
    if shape.face_count >= SIZE_PARAM and shape.face_count < SIZE_MAX:
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

    # TODO: perform preprocessing on the files gathered in files: string[]
    # Data set-up: file_path, vertex count, face count, label 
    preprocessed_files = []

    # Get label data
    labels = get_labels(classification_path) if classification_path is not None else None

    for file in files:
        if file is None or not exists(file): #NOTE: This is probably uneseccary, as we found the files on the filesystem earlier.
            log.error("The provided filepath %s does not exists", file)
            continue
        
        current_shape = Shape(file)
        current_shape.load()
        current_shape.find_aabb()

        # Find category if relevant
        if labels is not None:
            fnm = search(r'\d+', basename(file))

            if fnm is not None and fnm[0] in labels:
                current_shape.label = labels[fnm[0]]
                log.debug('Labeled mesh %s as %s', basename(file), current_shape.label)

        #In this case we just need to resave the mesh in the output path
        if not acceptable_size(current_shape):
            previous_f_count = current_shape.face_count
            previous_v_count = current_shape.vertex_count

            if current_shape.face_count > SIZE_MAX:
                current_shape.subsample_mesh(SIZE_PARAM)

                log.debug("Decimated shape %s, previously (%d) faces and (%d) vertices, currently (%d) faces and (%d) vertices", 
                            current_shape._get_name_from_path(), previous_f_count, previous_v_count, current_shape.face_count, current_shape.vertex_count)
                
            elif current_shape.face_count < SIZE_MIN:
                current_shape.supersample_mesh(SIZE_PARAM)

                log.debug("Supersampled shape %s, previously (%d) faces and (%d) vertices, currently (%d) faces and (%d) vertices", 
                            current_shape._get_name_from_path(), previous_f_count, previous_v_count, current_shape.face_count, current_shape.vertex_count)

        current_shape.find_aabb()
        current_shape.save_mesh_file(output_path) # Save the pre-processed shape into our database directory
        entry = current_shape.to_dict()

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