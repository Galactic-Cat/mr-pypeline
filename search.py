'''Module for comparing a single file against the database'''
from logging import getLogger
from os.path import basename, isfile
from typing import Dict
from re import sub

import numpy as np
from pandas import concat, DataFrame, read_csv, Series

from preprocess import single_preprocess

log = getLogger('search')

class Search:
    database: DataFrame

    def __init__(self, database_path: str):
        '''Initializes a new search class

        Args:
            database_path (str): Path to the database csv to use
        '''
        # Load database
        if not isfile(database_path):
            log.critical('Database path "%s" is not valid or inaccessible', database_path)
            return

        self.database = read_csv(database_path)

        if self.database is None:
            log.critical('Failed to retrieve database from "%s"', database_path)
            return

    def compare(self, path: str, classification_path: str = None, custom_label: str = None) -> DataFrame:
        '''Compares a file against the database

        Args:
            path (str): Path to the file to compare
            classification_path (str, optional): Path to a classification file for the file to compare. Defaults to None.
            custom_label (str, optional): A custom label for the file comparison

        Returns:
            DataFrame: A dataframe containing the distance to all features, and the appropriate file path and name, sorted from most to least alike
        '''
        # Check file path for validity
        if not isfile(path):
            log.critical('File path "%s" is not valid or inaccessible', path)
            return  

        # Extract the features from the mesh
        feature_map = single_preprocess(path, classification_path)
        
        if feature_map is None:
            log.critical('Failed to extract features from "%s" for comparison', path)
            return

        # Assign the custom label if provided
        if custom_label is not None:
            feature_map['label'] = custom_label

        # Define the column groups for use later
        distribution_feature_columns = ['A3', 'D1', 'D2', 'D3', 'D4']
        simple_feature_columns = ['aabb_volume', 'compactness', 'diameter', 'eccentricity', 'surface_area']

        # Get minimum and maximum values, first row is minimum, second row is maximum
        minmax_frame = DataFrame([self.database.min(), self.database.max()])[simple_feature_columns].rename(lambda i: ['min', 'max'][i])
        
        # Normalize the simple features and the entry to compare
        normalized = self.database[simple_feature_columns].apply(self.normalize_series)
        feature_map = self.normalize_entry(feature_map, minmax_frame)
        
        # Get the distances for the simple features, and the total distances
        simple_distances = self.simple_feature_distance(feature_map, normalized)
        distribution_distances = self.distribution_feature_distance(feature_map, self.database[distribution_feature_columns])
        total_distances = concat([simple_distances, distribution_distances, self.database['path']], axis=1)
        total_distances['name'] = total_distances['path'].apply(basename)
        total_distances['total_distance'] = total_distances[['simple_features', 'distribution_features']].apply(lambda s: np.sqrt(np.sum(s ** 2)), 1)
        
        return total_distances.sort_values(by='total_distance')

    def distribution_feature_distance(self, entry: Dict[str, float], values: DataFrame) -> DataFrame:
        '''Calculates the euclidian distance between the distribution features in the database and that of a single entry

        Args:
            entry (Dict[str, float]): The entry to calculate the distance for
            values (DataFrame): The other database entries to compare against

        Returns:
            DataFrame: A dataframe containing the distances to each value and their total euclidian distance (in column "distribution_features")
        '''
        # Define some anonymous functions to use
        to_dist = lambda a, b: np.sqrt(np.sum(abs(a - b) ** 2))
        to_numpy = lambda s: np.asarray([float(x) for x in sub(r'[\,\[\]]', '', s).split(' ')]) if type(s) is str else s

        # Convert strings to numpy arrays
        values = values[['A3', 'D1', 'D2', 'D3', 'D4']].apply(lambda s: s.apply(to_numpy))
        
        # Calculate the euclidian distance for each column
        values.transform(lambda c: c.apply(lambda a: to_dist(a, to_numpy(entry[c.name]))))
        
        # Calculate the euclidian distance for all distribution features per row
        values['distribution_features'] = self.normalize_series(values.apply(lambda s: np.sqrt(np.sum(s.apply(lambda v: np.sum(v ** 2)))), 1))

        return values

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
                if entry[c] is None:
                    entry[c] = np.nan
                entry[c] = (entry[c] - minmax.loc['min', c]) / (minmax.loc['max', c] - minmax.loc['min', c])

        return entry

    def normalize_series(value: Series) -> Series:
        '''Normalize the values of a Series to a range from zero to one

        Args:
            value (Series): The series to normalize

        Returns:
            Series: The normalized series
        '''
        maximum = value.max()
        minimum = value.min()

        return (value - minimum) / (maximum - minimum)

    def simple_feature_distance(self, entry: Dict[str, float], values: DataFrame) -> DataFrame:
        '''Calculates the euclidian distance between the simple features in the database and that of a single entry

        Args:
            entry (Dict[str, float]): The entry to calculate the distance for
            values (DataFrame): The other database entries to compare

        Returns:
            DataFrame: A dataframe containing the distances to each value and their total euclidian distance (in column "simple_features")
        '''
        values = values[['aabb_volume', 'compactness', 'diameter', 'eccentricity', 'surface_area']]
        distances = values.apply(lambda s: np.abs(s - entry[s.name])).fillna(0)
        distances['simple_features'] = self.normalize_series(distances.apply(lambda r: np.sqrt(np.sum(r ** 2)), 1))

        return distances
