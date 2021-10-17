'''Module for comparing a single file against the database'''
from logging import getLogger
from os.path import isfile
from typing import Dict, List, Tuple, Union

from numpy import ndarray, sqrt
from pandas import DataFrame, read_csv, Series

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
    
    # Get the entry and the database normalized using the minmax frame
    normalized = database.apply(lambda x: normalize_series(x, minmax_frame))
    normalized = normalized[normalized.notnull()]
    feature_map = normalize_entry(feature_map)
    
    # Get the distances for the simple features
    simple_distances = simple_feature_distance(feature_map, normalized)

def distribution_distance(entry: Dict[str, float], values: DataFrame) -> DataFrame:
    '''Calculates the euclidian distance between the distribution features in the database and that of a single entry

    Args:
        entry (Dict[str, float]): The entry to calculate the distance for
        values (DataFrame): The other database entries to compare against

    Returns:
        DataFrame: A dataframe containing the distances to each value and their total euclidian distance (in column "distribution_features")
    '''
    # TODO: Calculate distances from histograms
    raise NotImplementedError()

def normalize_entry(entry: Dict[str, float], minmax: DataFrame) -> Dict[str, float]:
    '''Normalizes a single entry using a minimum and maximum from a dataframe

    Args:
        entry (Dict[str, float]): The entry to normalize
        minmax (DataFrame): The dataframe containing the minimum and maximum values

    Returns:
        Dict[str, float]: The normalized entry
    '''
    keys = entry.keys()

    for c in minmax.columns:
        if c in keys:
            entry[c] = (entry[c] - minmax.loc['min', c]) / (minmax.loc['max', c] - minmax.loc['min', c])

    return entry

def normalize_series(value: Series, minmax: DataFrame) -> Series:
    '''Gets the normalized values for a series between a minimum and maximum

    Args:
        value (Series): The series to normalize
        minmax (DataFrame): The dataframe containing the minimum and maximum values

    Returns:
        Series: The normalized series
    '''
    if not value.name in minmax.columns:
        return None

    return (value - minmax.loc['min', value.name]) / (minmax.loc['max', value.name] - minmax.loc['min', value.name])

def simple_feature_distance(entry: Dict[str, float], values: DataFrame) -> DataFrame:
    '''Calculates the euclidian distance between the simple features in the database and that of a single entry

    Args:
        entry (Dict[str, float]): The entry to calculate the distance for
        values (DataFrame): The other database entries to compare

    Returns:
        DataFrame: A dataframe containing the distances to each value and their total euclidian distance (in column "simple_features")
    '''
    values = values[['aabb_volume', 'compactness', 'diameter', 'eccentricity', 'surface_area']]
    distances = values.apply(lambda s: abs(s - entry[s.name])).fillna(0)
    distances['simple_features'] = distances.apply(lambda r: sqrt(sum(r ** 2)))

    return distances
