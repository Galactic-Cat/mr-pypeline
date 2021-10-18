'''Module for comparing a single file against the database'''
import ast
from logging import getLogger
from os.path import isfile
from typing import Dict, List, Tuple, Union

import numpy as np
from pandas import DataFrame, read_csv, Series

from preprocess import single_preprocess
from util import converter

log = getLogger('search')

class SearchEngine():

    def __init__(self, db_path, classification_path = None):

        if not isfile(db_path):
            log.critical('Database path "%s" is not valid or inaccessible', db_path)
            return None
        # If we dont read with converters the csv returns the array as a string.
        self.database = read_csv(db_path, converters={'A3':converter, 'D1':converter, 'D2':converter, 'D3':converter, 'D4':converter})
        self.classification_data = classification_path


    def compare(self, file_path: str, custom_label: str = None) -> List[Tuple[str, str, float, Dict[str, float]]]:
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

        if self.database is None:
            log.critical('Database is not valid or inaccessible')
            return

        # Extract the features from the mesh
        feature_map = single_preprocess(file_path, self.classification_path)
        
        if feature_map is None:
            log.critical('Failed to extract features from "%s" for comparison', file_path)
            return

        # Assign the custom label if provided
        if custom_label is not None:
            feature_map['label'] = custom_label

        # Get minimum and maximum values, first row is minimum, second row is maximum
        minmax_frame = DataFrame([self.database.min(), self.database.max()])[['aabb_volume', 'compactness', 'diameter', 'eccentricity', 'surface_area']].rename(lambda i: ['min', 'max'][i])
        
        # Get the entry and the database normalized using the minmax frame
        normalized = self.database.apply(lambda x: self.normalize_series(x, minmax_frame))
        normalized = normalized[normalized.notnull()]
        feature_map = self.normalize_entry(feature_map)
        
        # Get the distances for the simple features
        simple_distances = self.simple_feature_distance(feature_map, normalized)

    def distribution_distance(self, entry: Dict[str, float], values: DataFrame) -> DataFrame:
        '''Calculates the euclidian distance between the distribution features in the database and that of a single entry

        Args:
            entry (Dict[str, float]): The entry to calculate the distance for
            values (DataFrame): The other database entries to compare against

        Returns:
            DataFrame: A dataframe containing the distances to each value and their total euclidian distance (in column "distribution_features")
        '''

        columns = ['A3', 'D1', 'D2', 'D3', 'D4']

        for c in columns:
            entry_dist = entry[c] 
            distances = []
            for other_entry in values[c]:
                value_dist = other_entry
                dist = np.linalg.norm(np.array(entry_dist) - np.array(value_dist))
                distances.append(dist)
            column_name = c + '_dist'
            values[column_name] =  np.array(distances)
        return values

    def normalize_entry(self, entry: Dict[str, float], minmax: DataFrame) -> Dict[str, float]:
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

    def normalize_series(self, value: Series, minmax: DataFrame) -> Series:
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

    def simple_feature_distance(self, entry: Dict[str, float], values: DataFrame) -> DataFrame:
        '''Calculates the euclidian distance between the simple features in the database and that of a single entry

        Args:
            entry (Dict[str, float]): The entry to calculate the distance for
            values (DataFrame): The other database entries to compare

        Returns:
            DataFrame: A dataframe containing the distances to each value and their total euclidian distance (in column "simple_features")
        '''
        values = values[['aabb_volume', 'compactness', 'diameter', 'eccentricity', 'surface_area']]
        distances = values.apply(lambda s: abs(s - entry[s.name])).fillna(0)
        distances['simple_features'] = distances.apply(lambda r: np.sqrt(sum(r ** 2)))

        return distances


# if __name__=='__main__':

#     eng  = SearchEngine('output/preprocess/database.csv')
#     entry = eng.database.iloc[0].to_dict()
#     database = eng.database
#     #values = database[['A3', 'D1' ,'D2' ,'D3' ,'D4']]
#     print(eng.distribution_distance(entry, database))
