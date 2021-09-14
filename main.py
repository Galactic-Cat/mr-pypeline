import open3d as o3d
import open3d.visualization.gui as gui
from MeshViewer import MainWindow

def main():

    gui.Application.instance.initialize()

    window = MainWindow(1024,768)
    window.load('./m100.off')
    gui.Application.instance.run()

if __name__ == "__main__":
    main()