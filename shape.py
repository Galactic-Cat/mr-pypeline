'''Module for finding and storing shape data'''

#from __future__ import annotations
from logging import getLogger
from os.path import basename, exists
from re import sub

from numpy import array, float64
from open3d import geometry, io, utility

database = None
log = getLogger('shape')

class Shape:
    '''Class representing a shape mesh and relevant info'''
    aabb: geometry.AxisAlignedBoundingBox = None # The minimal axis aligned bounding box
    db_name: str = None                          # The name of the file in the database
    face_count: int = None                       # The number of faces in this shape
    label: str = None                            # The shape class label of the Princeton and PBS labeled dataset
    loaded: bool = False                         # Stores whether or not the shape has been loaded from file
    mesh: geometry.TriangleMesh = None           # The triangle mesh if it's loaded
    vertex_count: int = None                     # The number of vertices in this shape

    def __init__(self, path: str, db_name: str = None):
        '''Constructs a new instance of the shape class

        Args:
            path (str): The path to load the shape from
            db_name (str, optional): The name in the in-memory database. Defaults to None.
        '''
        self.path = path

        if db_name is not None:
            log.error('Shape loading from dictionary not yet implemented') # TODO: load shape from database

    def find_aabb(self) -> None:
        '''Calculates the axis aligned minimal bounding box'''
        if self.mesh is None:
            log.debug('Need to load mesh to get the AABB, doing so')
            self.load()

        if self.mesh is None:
            log.error('Failed to load mesh, failed to get AABB')
            return

        self.aabb = self.mesh.get_axis_aligned_bounding_box()

    def load(self) -> None:
        '''Loads the shape data from path'''
        if self.path is None or not exists(self.path):
            log.error('No file at given path "%s"', self.path)
            return

        # Try to load the triangle mesh
        triangle_mesh = io.read_triangle_mesh(self.path)
        self.loaded = not triangle_mesh.is_empty()

        if not self.loaded:
            log.error('Failed to load triangle mesh from "%s"', self.path)
            return

        self.mesh = triangle_mesh
        log.debug('Loaded triangle mesh from file "%s"', self.path)

    def update_database(self, name: str = None) -> None:
        '''Updates the database with this shape

        Args:
            name (str, optional): The database name to use. Defaults to None.
        '''
        # Determine the name to use
        self.db_name = name or self.db_name or self._get_name_from_path()

        if self.db_name is None:
            log.error('Could not get a name to update database')
            return

        data = dict()

        if self.db_name in database:
            data = database[self.db_name]

        # Update data
        if self.aabb is not None:
            data['aabb_max'] = self.aabb.get_max_bound()
            data['aabb_min'] = self.aabb.get_min_bound()

        data['face_count'] = self.face_count or (data['face_count'] if 'face_count' in data else None)
        data['label'] = self.label or (data['label'] if 'label' in data else None)
        data['loads'] = self.loaded or (data['loads'] if 'loads' in data else None) # boolean and None OR
        data['path'] = self.path or (data['path'] if 'path' in data else None)
        data['vertex_count'] = self.vertex_count or (data['vertex_count'] if 'vertex_count' in data else None)

        # Update database
        database[self.db_name] = data

    def _get_name_from_path(self) -> str:
        '''Tries to get the filename from this shape's path

        Returns:
            str: The name found. None if no name could be found
        '''
        if self.path is None:
            return None

        split_base_name = basename(self.path).split('.')

        # Drop the extension if (likely) present
        if len(split_base_name) > 1:
            return '.'.join(split_base_name[:-1])

        return '.'.join(split_base_name)

    @classmethod
    def from_database(cls, name: str):# -> Shape:
        '''Generates a shape class instance from database data

        Args:
            name (str): The name of the shape to load

        Returns:
            Shape: The loaded shape class instanstance
        '''
        if database is None:
            log.error('Database has not been loaded in yet')
            return

        if not name in database:
            log.error('Could not find name "%s" in database')
            return

        data = database[name]
        shape = cls(data.path, name)
        aabb_vector = utility.Vector3dVector(array([data['aabb_max'], data['aabb_min']]))
        shape.aabb = geometry.AxisAlignedBoundingBox.create_from_points(aabb_vector)
        shape.face_count = data.face_count
        shape.label = data.label
        shape.vertex_count = data.vertex_count
        
        return shape

#Loads a full database
def load_database(path: str = None) -> None:
    '''Loads a database from a csv file, or creates a blank one if a path is omitted

    Args:
        path (str, optional): The path to load. Create a blank database if None. Defaults to None.
    '''
    global database

    if path is None:
        database = dict()
        return

    if not exists(path) or basename(path)[-4:] != '.csv':
        log.warning('Provided path "%s" does not exist or is not a csv file. Using a blank dictionary instead', path)
        database = dict()
        return

    keys = ['aabb_max', 'aabb_min', 'face_count', 'label', 'loads', 'path', 'vertex_count']
    database = dict()
    
    with open(path, 'r') as f:
        line = None

        while line != '':
            line = f.readline()
            entry_size = len(keys) + 1
            line_index = 0
            split_line = line.split(',')
            name = None

            for word in split_line:
                key_index = line_index % entry_size - 1
                line_index += 1
                word = word.strip()

                # New database entry
                if key_index < 0:
                    name = word
                    
                    if name in database:
                        log.warning('Name "%s" occurs multiple times, ignoring second occurence!', name)
                        name = None
                        continue

                    if name == '':
                        name = None
                        continue

                    database[name] = dict()
                    continue

                # Ignore data if name is not set (for duplicate names)
                if name is None or name == '':
                    continue

                # Add data to database entry
                key = keys[key_index]

                # Preserve Nones
                if word == '_none':
                    database[name][key] = None
                    continue

                # Make sure the value is typecast correctly
                if key in ['face_count', 'vertex_count']:
                    database[name][key] = int(word)
                elif key in ['loads']:
                    database[name][key] = bool(word)
                elif key in ['aabb_max', 'aabb_min']: # Special case, bounding box load numpy array
                    coords = word.split(' ')
                    database[name][key] = array(coords, dtype=float64)
                else:
                    database[name][key] = word

    log.debug('Loaded %d database entries', len(database))

#Saves the new 
def save_database(path: str) -> None:
    '''Saves the database to a CSV file

    Args:
        path (str): The path to the file to save to
    '''
    # Add CSV extension if missing
    if path[-4:] != '.csv':
        path = path + '.csv'

    keys = ['aabb_max', 'aabb_min', 'face_count', 'label', 'loads', 'path', 'vertex_count']
    entries = []

    for name in database:
        entry = [name]
        data = database[name]
        
        for key in keys:
            # Preserve Nones
            if not key in data or data[key] is None:
                entry.append('_none')
                continue
            # Representation of the array
            elif key in ['aabb_max', 'aabb_min']:
                entry.append(sub(r'(^\s+|\s\s+|\s+$)', ' ', str(data[key])[1:-1]).strip())
            # Other can just be stings
            else:
                entry.append(str(data[key]))

        entries.append(','.join(entry))

    with open(path, 'w') as f:
        f.writelines(entries)

    log.info('Saved %d database entries to "%s"', len(entries), path)