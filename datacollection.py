'''Module for preprocessing the off and ply files'''
from logging import getLogger
from os import listdir
from os.path import exists, isfile, isdir
import numpy as np
from shape import Shape
import pandas as pd

log = getLogger('data_collection')

dataframe = None

def visualize_data(output_path: str) -> None:
    ''' Function that will visualize the collected data given a .csv file

    Args:
        filepath (str): The path of the .csv file that collects information about the shapes in the database.
        output_file(str): The output folder in which to save the figures.

    '''
    if dataframe is None:
        log.error('Dataframe has not been instantiated.')
        return
    
    for c in dataframe.columns:
        ax = dataframe.hist(column = c,xrot = 90, legend = True)
        fig = ax[0][0].get_figure()
        fig.savefig(output_path + "/"+ c + '_hist.png')

    return

def calculate_features(output_path: str) -> None:
    
    if output_path is None:
        log.error('The outpath path "%s" does not exist.', output_path)
        return
    
    if dataframe is None:
        log.error('Dataframe has not been instantiated.')
        return

    column_list = ['vertex_count', 'face_count']
    for c in column_list:
        data = dataframe[c].to_numpy()
        
        with open(output_path + "/" + c + "_results.txt", 'w') as f:
            f.write("Mean:" + str(data.mean()) + '\n')
            f.write("25th Perc:" + str(np.percentile(data, 25)) + '\n')
            f.write("50th Perc:" + str(np.percentile(data, 50)) + '\n')
            f.write("75th Perc:" + str(np.percentile(data, 75)) + '\n')
            f.write("Min:" + str(data.min()) +'\n')
            f.write("Max:" + str(data.max()) +'\n')
            f.close()
    
    log.debug('Features have been calculated for the vertex and face count.')

    return 
 

def collect_shape_information(input_path: str, output_path: str) -> None:
    '''Function that preprocesses files from input to output

    Args:
        input (str): The input file/folder
        output (str): The output folder
    '''
    # Check if the input is valid
    if not exists(input_path):
        log.critical('Input path "%s" does not exist', input_path)
        return
    
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

    files_information = []

    for file in files:
        if file is None or not exists(file):
            log.error("The provided filepath %s does not exists", file)
            continue
        
        current_shape = Shape(file)
        current_shape.load()

        shape_information = {"face_count" : current_shape.face_count, "vertex_count" : current_shape.vertex_count}

        files_information.append(shape_information)

    global dataframe 
    
    dataframe = pd.DataFrame.from_dict(files_information)

    visualize_data(output_path)
    calculate_features(output_path)

    return

