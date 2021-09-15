'''Main program entrypoint'''

from open3d.visualization import gui

from arguments import args, setup_logger
from window import MainWindow

def main() -> None:
    '''The main function'''
    # Setup the loggers
    setup_logger(args.debug)

    # Initialize the application
    gui.Application.instance.initialize()

    # Create a window and load a file
    window = MainWindow(1024,768)

    window.load('./plane.ply')
    gui.Application.instance.run()

# Run the main code only if this is used as the entrypoint of the program
if __name__ == "__main__":
    main()