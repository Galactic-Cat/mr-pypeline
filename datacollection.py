'''Module for preprocessing the off and ply files'''
from logging import getLogger
from os.path import exists, isfile

from extraction import simple_features
from preprocess import calculate_mesh_center, find_aabb_points, convert_to_trimesh, single_preprocess
from util import locate_mesh_files

import trimesh as tm
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
import ast


log = getLogger('data_collection')

dataframe = None

# def verify_basic_features(output_path) -> None:
#     mesh = io.read_triangle_mesh("output\preprocess\m741\m741.off")
#     entry = simple_features(mesh)
#     print(entry)

#     mesh = io.read_triangle_mesh("output\preprocess\m517\m517.off")
#     entry = simple_features(mesh)
#     print(entry)
#     return

def visualize_data(data: np.array, feature_name: str, output_path: str, title: str, xlabel:str, ylabel:str = '% of Shapes', bins_: int = 15) -> None:
    ''' Function that will visualize the collected data given a .csv file

    Args:
        filepath (str): The path of the .csv file that collects information about the shapes in the database.
        output_file(str): The output folder in which to save the figures.

    '''
    if dataframe is None:
        log.error('Dataframe has not been instantiated.')
        return

    # if feature_name == 'face_count' or feature_name == 'vertex_count':
    #     data_threshold = np.percentile(data, 85)
    #     data = data[(data <= data_threshold)]

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

def verify_translation(current_mesh: tm.Trimesh) -> float:

    centroid = calculate_mesh_center(current_mesh)
    norm = round(np.linalg.norm(centroid),3)
    return norm


def display_class_distributions(input_path:str, output_path: str) -> None:
    
    df = pd.read_csv(input_path)
    df = df.drop(columns = ['aabb_max','aabb_min','path', 'face_count', 'vertex_count', 'surface_area', 'aabb_volume', 'compactness', 'diameter', 'eccentricity'])
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
                axis='x',
                which='both',      
                bottom=False,      
                top=False,         
                labelbottom=False) 
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

def verify_scaling(current_mesh: tm.Trimesh) -> float:

    min, max = find_aabb_points(current_mesh)
    return round(np.max(np.abs(max) + np.abs(min)), 4)
 
def verify_rotation(current_mesh: tm.Trimesh) -> float:
    pca = current_mesh.principal_inertia_vectors
    return abs(pca[0][0])

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
            
            current_mesh = tm.load(file)
            aabb_size = verify_scaling(current_mesh)
            x_coordinate = verify_rotation(current_mesh)
            distance_from_center = verify_translation(current_mesh)
            shape_information = {"face_count" :current_mesh.faces.shape[0], "vertex_count" : current_mesh.vertices.shape[0], 
                                    "aabb_size": aabb_size, "centroid": distance_from_center,"x_coord": x_coordinate}

            files_information.append(shape_information)

        global dataframe 
        
        dataframe = pd.DataFrame.from_dict(files_information)
        visualize_data(data = dataframe['face_count'].to_numpy(), title = 'Distribution of Face counts', feature_name = 'face_count', output_path = output_path, xlabel = 'Face Count')
        visualize_data(data = dataframe['vertex_count'].to_numpy(), title = 'Distribution of Vertex counts' , feature_name =  'vertex_count', output_path = output_path, xlabel = 'Vertex Count')
        visualize_data(data = dataframe['aabb_size'].to_numpy(), title = 'Distribution of AABB sizes', feature_name = 'aabb_size', output_path = output_path, xlabel ='AABB Size')
        visualize_data(data = dataframe['centroid'].to_numpy(), title = 'Distribution of centroid distance to origin', feature_name = 'centroid', output_path = output_path, xlabel = 'Norm of centroid')
        visualize_data(data = dataframe['x_coord'].to_numpy(), title = 'Distribution of X-coordinate alignments', feature_name = 'x_coord', output_path = output_path, xlabel = 'Absolute x-coord of major eigenvector')
        calculate_features(output_path)
        #verify_basic_features(output_path)

    return

# if __name__ == '__main__':
#      mesh = tm.load('./test_shapes/m100/m100.off')
#      print(verify_rotation(mesh))
#      print(verify_translation(mesh))
#      print(verify_scaling(mesh))
