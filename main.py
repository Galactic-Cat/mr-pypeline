'''Main program entrypoint'''
from arguments import args, setup_logger
from logging import getLogger

def run_preprocess() -> None:
    '''Runs the preprocessing part of the program'''
    # Late import to prevent loading unnecessary packages
    from preprocess import preprocess
    preprocess(args.input[0], args.output[0])

def run_view() -> None:
    '''Runs the viewing part of the program'''
    # Late import to prevent loading unnecessary packages
    from open3d.visualization import gui
    from window import MainWindow

    # Initialize the application
    gui.Application.instance.initialize()

    # Create a window and load a file
    window = MainWindow(1024,768)

    window.load('./plane.ply')
    gui.Application.instance.run()

mode_table = {
    None: run_view,
    'preprocess': run_preprocess,
    'view': run_view
}

# Run the main code only if this is used as the entrypoint of the program
if __name__ == "__main__":
    setup_logger(args.debug)
    getLogger('arguments').debug('Selected mode is %s', args.mode or 'view')
    mode_table[args.mode]()
