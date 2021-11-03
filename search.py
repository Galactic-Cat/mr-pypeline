'''Module for comparing a single file against the database'''
from logging import getLogger
from os.path import basename, isfile
from typing import Dict, Union
from re import sub

from annoy import AnnoyIndex
import numpy as np
from pandas import concat, DataFrame, read_csv, Series
from pyemd import emd

from preprocess import single_preprocess

log = getLogger('search')
scalar_columns = ['surface_area', 'compactness', 'aabb_volume', 'diameter', 'eccentricity']
distribution_columns = ['A3', 'D1', 'D2', 'D3', 'D4']

class Search:
    approximated_nearest_neighbours: AnnoyIndex = None
    averages: Series = None
    database: DataFrame = None
    raw_database: DataFrame = None
    stddevs: Series = None
    search_sample = 5

    def __init__(self, database_path: str):
        '''Initializes a new search class

        Args:
            database_path (str): Path to the database csv to use
        '''
        # Load database
        if not isfile(database_path):
            log.critical('Database path "%s" is not valid or inaccessible', database_path)
            return

        self.raw_database = read_csv(database_path)

        if self.raw_database is None:
            log.critical('Failed to retrieve database from "%s"', database_path)
            return

        self.standardize_database()
        self.prepare_nearest_neighbours()

    def compare(self, path: str, classification_path: str = None, custom_label: str = None, use_ann: bool = True) -> DataFrame:
        '''Compares a file against the database

        Args:
            path (str): Path to the file to compare
            classification_path (str, optional): Path to a classification file for the file to compare. Defaults to None.
            custom_label (str, optional): A custom label for the file comparison. Defaults to None.
            use_ann (bool, optional): Use Approximate Nearest Neighbours to search. Defaults to True.

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

        feature_series = self.prepare_entry(feature_map).fillna(0.0)

        # Assign the custom label if provided
        if custom_label is not None:
            feature_map['label'] = custom_label
        
        if use_ann:
            order, distances = self.approximated_nearest_neighbours.get_nns_by_vector(vector = create_feature_vector(feature_series), n = self.database.shape[0], include_distances=True)#self.search_sample, include_distances=True)
            total_distances = self.raw_database.loc[order, 'path']
            names = total_distances.apply(basename)
            names.name = 'name'
            total_distances = concat([total_distances, names], axis=1)
            total_distances['total_distance'] = distances
            
            return total_distances
        else:
            # Get the distances for the simple features, distribution features, and the total distances
            simple_distances = self.scalar_feature_distance(feature_series)
            distribution_distances = self.distribution_feature_distance(feature_series)
            total_distances = concat([simple_distances, distribution_distances, self.raw_database['path']], axis=1)
            total_distances['name'] = total_distances['path'].apply(basename)
            total_distances['total_distance'] = total_distances[['scalar_features', 'distribution_features']].apply(lambda s: np.sqrt(np.sum(s ** 2)), axis=1)
            
            return total_distances.sort_values(by='total_distance')

    def distribution_feature_distance(self, entry: Series) -> DataFrame:
        '''Calculates the euclidian distance between the distribution features in the database and that of a single entry

        Args:
            entry (Series): The entry to calculate the distance for

        Returns:
            DataFrame: A dataframe containing the standardized distances to each value and their total euclidian distance (in column "distribution_features")
        '''
        # Calculate the EMD for each feature for each value
        distance_matrix = generate_distance_matrix(20, 0.5)
        values = self.database[distribution_columns].apply(lambda c: c.apply(lambda a: emd(a, entry[c.name], distance_matrix)))
        
        # Standardize the calculated distances
        averages = values.mean()
        stddevs = values.std()
        
        values.transform(lambda c: (c - averages) / stddevs, axis=1)
        
        # Calculate the euclidian distance for all distribution features per row
        #values['distribution_features'] = values.apply(lambda r: np.sqrt(np.sum(r ** 2)), axis=1)

        return values

    def scalar_feature_distance(self, entry: Series) -> DataFrame:
        '''Calculates the euclidian distance between the scalar features in the database and that of a single entry

        Args:
            entry (Series): The entry to calculate the distance for

        Returns:
            DataFrame: A dataframe containing the distances to each value and their total euclidian distance (in column "scalar_features")
        '''
        distances = self.database[scalar_columns].apply(lambda s: np.abs(s - entry[s.name])).fillna(0)
        averages = distances.mean()
        stddevs = distances.std()
        
        distances.transform(lambda c: (c - averages) / stddevs, axis=1)

        distances['scalar_features'] = distances.apply(lambda r: np.sqrt(np.sum(r ** 2)), axis=1)

        return distances

    def prepare_entry(self, entry: Dict[str, Union[float, np.ndarray]]) -> Series:
        '''Standardizes an entry dictionary and creates a feature vector from it

        Args:
            entry (Dict[str, Union[float, np.ndarray]]): The entry to standardize and vectorize

        Returns:
            Series: The dictionary as a feature vector
        '''
        raw_series = Series(entry)
        scalars = raw_series[scalar_columns].fillna(0.0)
        distributions = raw_series[distribution_columns].apply(np.asarray)

        return concat([scalars, distributions])

    def prepare_nearest_neighbours(self, trees: int = 7) -> None:
        '''Prepares a NN forest for quick search

        Args:
            trees (int, optional): The number of trees to generated. Defaults to 7.
        '''
        bins = self.database['A3'][0].shape[0]
        self.approximated_nearest_neighbours = AnnoyIndex(len(scalar_columns) + len(distribution_columns) * bins, metric='euclidean')
        feature_vectors = self.database[scalar_columns + distribution_columns].apply(create_feature_vector, axis=1)

        for i, fv in enumerate(feature_vectors):
            self.approximated_nearest_neighbours.add_item(i, fv)
        
        self.approximated_nearest_neighbours.build(trees)

    def standardize_database(self) -> None:
        '''Standardizes the loaded database's feature values'''
        # Standardize the scalar features
        scalars = self.raw_database[scalar_columns]
        means = np.mean(scalars)
        stddevs = np.std(scalars)
        scalars.transform(lambda row: (row - means) / stddevs, axis=1)

        # Convert the (already normalized) serialized histograms to numpy array
        to_numpy = lambda s: np.asarray([float(x) for x in sub(r'[\,\[\]]', '', s).split(' ')]) if type(s) is str else s
        distributions = self.raw_database[distribution_columns].apply(lambda s: s.apply(to_numpy))

        # Store the result
        self.database = concat([scalars, distributions], axis=1).fillna(0.0)

def generate_distance_matrix(size: int, unit_distance: float = 1.0) -> np.ndarray:
    '''Generates a distance matrix for pyemd

    Args:
        size (int): The size of the distance matrix, such that the matrix is size wide, and size high
        unit_distance (float, optional): The distance per unit, such that the distance between bin 1 and 2 is this. Defaults to 1.0.

    Returns:
        np.ndarray: A 2D distance matrix
    '''
    matrix = np.zeros((size, size))

    for i in range(size):
        for j in range(i, size):
            matrix[i,j] = matrix[j,i] = abs(i - j) * unit_distance

    return matrix

def create_feature_vector(row: Series) -> np.ndarray:
    '''Turns a database row into a numpy array

    Args:
        row (Series): The row to transform

    Returns:
        np.ndarray: The created vector
    '''
    return np.concatenate([row[scalar_columns], row['A3'], row['D1'], row['D2'], row['D3'], row['D4']])

if __name__ == '__main__':
    s = Search('./output/preprocess/database.csv')
    print(s.compare('./test_shapes/m100.off'))
