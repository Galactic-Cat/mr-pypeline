'''Module for preprocessing the off and ply files'''
from collections import defaultdict
from logging import getLogger
from os.path import exists, isfile

from extraction import simple_features
from preprocess import calculate_mesh_center, find_aabb_points
from search import Search
from statistics import mode
from util import locate_mesh_files

import trimesh as tm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast


log = getLogger('data_collection')

dataframe = None

def verify_basic_features() -> None:
    mesh = tm.load("output/preprocess/360/360.off")
    entry = simple_features(mesh)
    print("Bearing information")
    print(entry)

    mesh = tm.load("output/preprocess/22/22.off")
    entry = simple_features(mesh)
    print("Cup information:")
    print(entry)
    return

def visualize_data(data: np.array, feature_name: str, output_path: str, title: str, xlabel:str, ylabel:str = '% of Shapes', bins_: int = 15, xticks = None, xtick_labels =  None) -> None:
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
    n, bins, patches = ax.hist(data, bins = bins_, weights=np.zeros_like(data) + 100. / data.size,  ec="k")
    ax.set_xlabel(xlabel, size=15)
    ax.set_ylabel(ylabel, size=15)
    ax.set_title(title)
    if xticks:
        ax.set_xticks(xticks)
    if xtick_labels:
        ax.set_xticklabels(xtick_labels)
    ax.legend
    ax.set_ylim([0,100])
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
            plt.xticks(range(0, 31, 3), [round (x,2) for x in np.arange(0, 1.01, (3/30))])
            plt.ylim([0,0.20])
            if column == 'D3':
                plt.xlabel(f'Relative {column} Area')
            elif column == 'D4':
                plt.xlabel(f'Relative {column} Volume')
            else:
                plt.xlabel(f'Relative {column} Distance')
            plt.ylabel('Frequency')
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

def verify_flip_testing(current_mesh: tm.Trimesh):
    f_sign = [0,0,0]

    vertices =  current_mesh.vertices

    for face in current_mesh.faces:

        vertex_total = np.zeros(3,dtype=float)

        for vertex in face:
            vertex_total+=vertices[vertex]

        center_of_face = vertex_total/3

        for i,coord in enumerate(center_of_face):
            if coord >= 0:
                f_sign[i] += 1
            else:
                f_sign[i] -= 1

    f_sign = [1 if r >= 0 else -1 for i,r in enumerate(f_sign)]

    return f_sign[0]

def verify_scaling(current_mesh: tm.Trimesh) -> float:

    min, max = find_aabb_points(current_mesh)
    return round(np.max(np.abs(max) + np.abs(min)), 4)
 
def verify_rotation(current_mesh: tm.Trimesh) -> float:
    return abs(current_mesh.principal_inertia_vectors[0][2]) / np.max(np.abs(current_mesh.principal_inertia_vectors[0]))

def verify_watertightness(current_mesh: tm.Trimesh):
    if current_mesh.is_watertight:
        return 1
    return 0

def test_flipping(input_path: str):
    files = locate_mesh_files(input_path)
        
    log.debug('Found %d files to preprocess', len(files))

    flipping_list = []

    for file in files:
        if file is None or not exists(file):
            log.error("The provided filepath %s does not exists", file)
            continue
            
        current_mesh = tm.load(file)
        flip_testing = verify_flip_testing(current_mesh)
        flipping_list.append(flip_testing)
    print(flipping_list)

def create_confusion_matrix(labels: list, dictionary: dict, method:str = 'ANN'):
    
    conf_matrix = np.zeros(shape=(len(labels), len(labels)))

    df = pd.DataFrame(conf_matrix, columns= labels, index = labels)

    for label in labels:
    
        for key in dictionary[label].keys():
            df.at[label,key] = dictionary[label][key]
            
    ax = plt.axes()
    sns.heatmap(df, annot= True, vmin = 0, vmax = 134)

    ax.set_title(method + "Confusion Matrix")
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Ground Truth')
    plt.show()

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
            z_coordinate = verify_rotation(current_mesh)
            distance_from_center = verify_translation(current_mesh)
            watertight = verify_watertightness(current_mesh)
            flip = verify_flip_testing(current_mesh)
            shape_information = {"face_count" :current_mesh.faces.shape[0], "vertex_count" : current_mesh.vertices.shape[0], 
                                    "aabb_size": aabb_size, "centroid": distance_from_center,"z_coord": z_coordinate, "water_tightness":watertight, "flip_testing": flip}
            files_information.append(shape_information)

        global dataframe 
        
        dataframe = pd.DataFrame.from_dict(files_information)
        visualize_data(data = dataframe['face_count'].to_numpy(), title = 'Distribution of Face counts', feature_name = 'face_count', output_path = output_path, xlabel = 'Face Count')
        visualize_data(data = dataframe['vertex_count'].to_numpy(), title = 'Distribution of Vertex counts' , feature_name =  'vertex_count', output_path = output_path, xlabel = 'Vertex Count')
        visualize_data(data = dataframe['aabb_size'].to_numpy(), title = 'Distribution of AABB sizes', feature_name = 'aabb_size', output_path = output_path, xlabel ='AABB Size')
        visualize_data(data = dataframe['centroid'].to_numpy(), title = 'Distribution of centroid distance to origin', feature_name = 'centroid', output_path = output_path, xlabel = 'Norm of centroid')
        visualize_data(data = dataframe['z_coord'].to_numpy(), title = 'Distribution of Z-coordinate alignments', feature_name = 'z_coord', output_path = output_path, xlabel = 'Absolute z-coord of major eigenvector')
        visualize_data(data = dataframe['water_tightness'].to_numpy(), title = 'Distribution of watertightness', feature_name = 'water_tightness', output_path = output_path, xlabel = 'Water tightness', bins_ = [-.5,.5,1.5], xticks= (0,1), xtick_labels = ['Non-watertight', 'Watertight'])
        visualize_data(data = dataframe['flip_testing'].to_numpy(), title = 'Distribution of x-coord sign', feature_name = 'flip_testing', output_path = output_path, xlabel = 'Flip test', bins_ = [-1.25,0,1.25], xticks= (-1,1), xtick_labels = ['Negative', 'Positive'])
        calculate_features(output_path)

    return

def collect_precision_recall(input_path: str) -> defaultdict():
    '''Function that collect precision and recall using the pipeline and
    preprocessed database.

    Args:
        input (str): The input file/folder
    '''

    files = locate_mesh_files(input_path)
    
    s = Search(input_path + '/database.csv')

    labels = list(set(s.raw_database['label'].values))

    database_results = defaultdict(lambda: defaultdict(float))

    precision_dict = defaultdict(float)

    recall_dict = defaultdict(float)

    overall_precision = 0.0

    overall_recall = 0.0

    for file in files:

        file_label = s.raw_database[s.raw_database['path'] == file].label.values[0]

        class_size = len(s.raw_database[s.raw_database['label'] == file_label])

        compare_results = s.compare(file, preprocess = False, use_ann=False)
        
        results = compare_results['label'].to_list()[0:7]

        TP = 0
        FP = 0

        for q_label in results:
            database_results[file_label][q_label] +=1
            if q_label == file_label:
                TP +=1
            else:
                FP +=1
        
        FN = class_size - TP
        precision = TP/(TP + FP)
        recall = TP/(TP + FN)

        precision_dict[file_label] += precision
        recall_dict[file_label] += recall

        overall_precision += precision
        overall_recall += recall
    
    for label in labels:
        class_size = len(s.raw_database[s.raw_database['label'] == label])
        precision_dict[label] /= class_size
        recall_dict[label] /= class_size

    overall_precision/= len(s.raw_database)
    overall_recall/= len(s.raw_database)

    create_confusion_matrix(labels, database_results)

    return database_results, precision_dict, recall_dict, overall_precision, overall_recall


def collect_MAP(input_path: str, ann:bool = False):
    '''Function that collects the MAP of the pipeline

    Args:
        input (str): The input file/folder
        ann (bool): Whether to use ANN or not
    '''

    files = locate_mesh_files(input_path)
    
    s = Search(input_path + '/database.csv')

    precision_class_dict = defaultdict(lambda: defaultdict(float))

    for i,file in enumerate(files):
    
        file_label = s.raw_database[s.raw_database['path'] == file].label.values[0]

        class_size = len(s.raw_database[s.raw_database['label'] == file_label])

        compare_results = s.compare(file, preprocess = False, use_ann=ann)
        
        results = compare_results['label'].to_list()
        
        for k in range(1,20):
            result_subset = results[0:k]
            TP = 0
            FP = 0

            for q_label in result_subset:
                if q_label == file_label:
                    TP +=1
                else:
                    FP +=1
                    
            #Precision calculated for file with k results
            precision = TP/(TP + FP)

            precision_class_dict[file_label][k] += precision/class_size
        
    map_class_results = defaultdict(float)
    
    for label in precision_class_dict.keys():
        dictionary_k =  precision_class_dict[label]
        map = 0
        for k in dictionary_k.keys():
            map += dictionary_k[k]/19
        
        map_class_results[label] = map 

    map_total = sum(list(map_class_results.values()))/len(map_class_results)

    return map_class_results, map_total


def plot_grouped_hist(labels, ann_results, custom_results, y_label):
    x = np.arange(len(labels))
    
    ann_results = [ann_results[key] for key in labels]
    custom_results = [custom_results[key] for key in labels]
    
    width = 0.4
    fig, ax = plt.subplots()
    
    ann_rects = ax.bar(x - width/2, ann_results, width, label = 'ANN')
    custome_rects = ax.bar(x + width/2, custom_results, width, label = 'Custom')
    
    ax.set_ylabel(y_label)
    ax.set_title(y_label + 'by search method')
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation = 90)
    ax.legend()

    fig.tight_layout()

    plt.show()
