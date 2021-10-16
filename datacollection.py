'''Module for preprocessing the off and ply files'''
from logging import getLogger
from os.path import exists, isfile
from open3d import io, geometry
from preprocess import find_aabb_points
from util import compute_pca, locate_mesh_files

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast


log = getLogger('data_collection')

dataframe = None

def visualize_data(data: np.array, feature_name: str, output_path: str, title: str, xlabel:str, ylabel:str = '% of Shapes', bins_: int = 15) -> None:
    ''' Function that will visualize the collected data given a .csv file

    Args:
        filepath (str): The path of the .csv file that collects information about the shapes in the database.
        output_file(str): The output folder in which to save the figures.

    '''
    if dataframe is None:
        log.error('Dataframe has not been instantiated.')
        return
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    n, bins, patches = ax.hist(data, bins = bins_, weights=np.zeros_like(data) + 100. / data.size)
    ax.set_xlabel(xlabel, size=15)
    ax.set_ylabel(ylabel, size=15)
    ax.set_title(title)
    ax.legend
    plt.savefig(output_path + "/"+ feature_name + '_hist.png')
    plt.close()

    return

def display_class_distributions(input_path:str, output_path: str) -> None:
    
    df = pd.read_csv(input_path)
    df = df.drop(columns = ['path', 'face_count', 'vertex_count', 'surface_area', 'aabb_volume', 'compactness', 'diameter', 'eccentricity'])
    df = df.dropna()
    
    labels = df['label'].dropna().unique()

    for c_label in labels:
        sub_set = df.loc[df['label']==c_label]
        #sub_set = df.drop(columns = 'label')

        for i, column in enumerate(sub_set.columns):
            if column == 'label':
                continue
            entries = sub_set[column]
            for entry in entries:
                entry = ast.literal_eval(entry)
                bins = [i for i in range(0,len(entry))]
                plt.plot(bins, entry)

            plt.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False) # labels along the bottom edge are off
            plt.title(f'{column} Distribution for class: {c_label}')
            plt.savefig(output_path + f'/{c_label}_{column}_dist.png')
            plt.close()

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

def verify_scaling(current_mesh: geometry.TriangleMesh) -> float:

    min, max = find_aabb_points(current_mesh)
    return round(np.max(np.abs(max) + np.abs(min)), 4)
 
def verify_rotation(current_mesh: geometry.TriangleMesh):
    x, _, _, _ = compute_pca(current_mesh)
    return abs(x[0])


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

    if isfile(input_path):
        display_class_distributions(input_path, output_path)

    else:
        
        files = locate_mesh_files(input_path)
        
        log.debug('Found %d files to preprocess', len(files))

        files_information = []

        for file in files:
            if file is None or not exists(file):
                log.error("The provided filepath %s does not exists", file)
                continue
            
            current_mesh = io.read_triangle_mesh(file)
            aabb_size = verify_scaling(current_mesh)
            x_coordinate = verify_rotation(current_mesh)
            shape_information = {"face_count" : len(current_mesh.triangles), "vertex_count" : len(current_mesh.vertices), 
                                    "aabb_size": aabb_size, "x_coord": x_coordinate}

            files_information.append(shape_information)

        global dataframe 
        
        dataframe = pd.DataFrame.from_dict(files_information)
        visualize_data(data = dataframe['face_count'].to_numpy(), title = 'Distribution of Face counts', feature_name = 'face_count', output_path = output_path, xlabel = 'Face Count')
        visualize_data(data = dataframe['vertex_count'].to_numpy(), title = 'Distribution of Vertex counts' , feature_name =  'vertex_count', output_path = output_path, xlabel = 'Vertex Count')
        visualize_data(data = dataframe['aabb_size'].to_numpy(), title = 'Distribution of AABB sizes', feature_name = 'aabb_size', output_path = output_path, xlabel ='AABB Size')
        visualize_data(data = dataframe['x_coord'].to_numpy(), title = 'Distribution of X-coordinate alignments', feature_name = 'x_coord', output_path = output_path, xlabel = 'Absolute x-coord of major eigenvector')
        calculate_features(output_path)

    return

#TODO COLLECT NORMALS FOR THE BARY CENTER BEFORE AND AFTER AND CHECK THE HISTOGRAMS

