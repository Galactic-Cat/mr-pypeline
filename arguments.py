'''Retrieves any command line arguments and creates a logger'''

from argparse import ArgumentParser, HelpFormatter, Namespace
from datetime import datetime
from logging import FileHandler, Formatter, getLogger, StreamHandler
from os import mkdir, path
from sys import exit as sys_exit, stdout

def setup_logger(stream_level: str) -> None:
    '''Sets up a file and stream logger
    Args:
        stream_level (str): The logging level for the stream handler
    '''
    # Create a log directory if none exists
    if not path.isdir('./logs'):
        mkdir('./logs')

    # Create the logger and the file and stream handlers.
    logger = getLogger()
    starttime = datetime.now().strftime('%y.%m.%d_%H.%M')
    file = FileHandler('./logs/' + starttime + '.log', 'a')
    file_formatter = Formatter('%(asctime)16s | %(levelname)-8s | %(name)-25s >> %(message)s', datefmt='%y/%m/%d - %H:%M')
    stream = StreamHandler(stdout)
    stream_formatter = Formatter('%(asctime)5s | %(levelname)-8s | %(name)-10s >> %(message)s', datefmt='%H:%M')

    # Link the file and stream handlers to the base logger
    file.setLevel(10)
    file.setFormatter(file_formatter)
    stream.setLevel(stream_level.upper())
    stream.setFormatter(stream_formatter)
    logger.addHandler(file)
    logger.addHandler(stream)
    logger.setLevel(10)

    # Silence matplotlib logger debug and info messages
    mpl = getLogger('matplotlib')
    mpl.setLevel(30)

def parse_arguments() -> Namespace:
    '''Creates an argument parser and parses the arguments
    Returns:
        Namespace: The parsed arguments as a argparse namespace
    '''
    # Create the argument parser
    parser = ArgumentParser(formatter_class=lambda prog: HelpFormatter(prog, max_help_position=65))

    parser.add_argument('--debug', '-d', metavar='CHOICE', nargs='?', choices=[
        'debug',
        'info',
        'warning',
        'error',
        'critical'
    ], default='info', const='debug', help='The verbosity of the program in the console.')

    # Parse the arguments
    return parser.parse_args()

# Prevent running this file on its own
if __name__ == '__main__':
    print('arguments.py should not be run as the main application, use main.py instead')
    sys_exit(1)

# Create a global variable containing the parsed arguments
args = parse_arguments()
