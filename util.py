'''Utility functions'''

from typing import Iterable, Tuple


def grouped(iterable: Iterable, count: int) -> Tuple[Iterable]:
    '''Returns the iterable zipped into groups of a specified count

    Args:
        iterable (Iterable): The iterable to group
        count (int): The size of the groups (if available)

    Returns:
        Tuple[Iterable]: The count-tuple of iterable list
    '''
    return zip(*[iter(iterable)]*count)