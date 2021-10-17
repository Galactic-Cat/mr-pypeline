'''Module for comparing a single file against the database'''
from logging import getLogger
from os.path import isfile
from typing import Dict, List, Tuple, Union

from numpy import ndarray
from pandas import DataFrame, read_csv

from preprocess import single_preprocess

log = getLogger('search')

def compare(file_path: str, database_path: str, classification_path: str = None, custom_label: str = None) -> List[Tuple[str, str, float, Dict[str, float]]]:
    '''Compares a file against the database

    Args:
        file_path (str): Path to the file to compare
        database_path (str): Path to the database csv file to compare against
        classification_path (str, optional): Path to a classification file for the file to compare. Defaults to None.
        custom_label (str, optional): A custom label for the file comparison

    Returns:
        List[Tuple[str, str, float, Dict[str, float]]]: A list of similar file in order of alikeness.
            The files are tuples giving: filename, file path, overall alikeness, alikness for each feature
    '''
    # Check file paths for validity
    if not isfile(file_path):
        log.critical('File path "%s" is not valid or inaccessible', file_path)
        return

    if not isfile(database_path):
        log.critical('Database path "%s" is not valid or inaccessible', database_path)
        return

    # Read the database
    database = read_csv(database_path)

    if database is None:
        log.critical('Failed to retrieve database from "%s"', database_path)
        return

    # Extract the features from the mesh
    feature_map = single_preprocess(file_path, classification_path)
    
    if feature_map is None:
        log.critical('Failed to extract features from "%s" for comparison', file_path)
        return

    # Assign the custom label if provided
    if custom_label is not None:
        feature_map['label'] = custom_label

    # Get minimum and maximum values, first row is minimum, second row is maximum
    minmax_frame = DataFrame([database.min(), database.max()])[['aabb_volume', 'compactness', 'diameter', 'eccentricity', 'surface_area']].rename(lambda i: ['min', 'max'][i])

    # TODO: Compare the entry to all other entries, produce numerical values differences etc, using apply probably
