'''Module for preprocessing the off and ply files'''
from logging import getLogger
from os import listdir, mkdir
from os.path import exists, isfile, isdir
from shape import Shape

log = getLogger('preprocess')
SIZE_PARAM = 1000 #Random value for now


def acceptable_size(shape: Shape) -> bool:

    max_size = SIZE_PARAM + int(SIZE_PARAM * 0.1)
    min_size = SIZE_PARAM - int(SIZE_PARAM * 0.1)
    if shape.vertex_count <= SIZE_PARAM and shape.vertex_count > min_size:
        return True
    
    if shape.vertex_count >= SIZE_PARAM and shape.vertex_count < max_size:
        return True

    return False

def preprocess(input_path: str, output_path: str) -> None:
    '''Function that preprocesses files from input to output

    Args:
        input (str): The input file/folder
        output (str): The output folder
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

    for file in files:
        if file is None or not exists(file):
            log.error("The provided filepath %s does not exists", file)
            continue
        
        current_shape = Shape(file)
        current_shape.load()

        if acceptable_size(current_shape):
            #save the exact same file in the output directory
            pass
        else:
            #Call preprocessing functions here
            #Give shape new "path"
            pass



    return

