'''Main Window class'''

from logging import getLogger
from os.path import exists

from open3d import io
from open3d.visualization import gui, rendering

class MainWindow():
    '''Class for drawing the main window'''

    log = getLogger('MainWindow')

    def __init__(self, width: int, height: int) -> None:
        '''Initializes a MainWindow instance with a predefined height and width

        Args:
            width (int): The width of the window
            height (int): The height of the window
        '''
        self.window = gui.Application.instance.create_window("MR-pypeline", width, height)

        self.create_3D_scene()

        # TODO: add 3d scene to window

        font_size = self.window.theme.font_size
        margin_size = int(0.5 * font_size)
        element_sep = int(1.5 * font_size)

        # Create options pannel
        self._options_panel = gui.Vert(0, gui.Margins(margin_size, margin_size, margin_size, margin_size))

        # Create buttons for loading pannel
        self._load_mesh_button = gui.Button("Add Mesh")
        self._load_mesh_button.horizontal_padding_em = 0.2
        self._load_mesh_button.vertical_padding_em = 0.1
        self._load_mesh_button.set_on_clicked(self._open_loading_dialog) # TODO

        self._clear_mesh_button = gui.Button("Clear Mesh")
        self._clear_mesh_button.horizontal_padding_em = 0.2
        self._clear_mesh_button.vertical_padding_em = 0.1
        self._clear_mesh_button.set_on_clicked(self._clear_scene) # TODO

        # Create layout inside options panel
        h_grid = gui.Horiz(0.25 * font_size)
        h_grid.add_stretch()
        h_grid.add_child(self._load_mesh_button)
        h_grid.add_child(self._clear_mesh_button)
        h_grid.add_stretch()
        self._options_panel.add_child(h_grid)

        self.window.add_child(self._scene_3d)
        self.window.add_child(self._options_panel)

    def load(self, filepath: str) -> bool:
        '''Attempts to load a file from a path into the viewport

        Args:
            filepath (str): The file path

        Returns:
            bool: Whether the loading succeeded
        '''
        if not exists(filepath):
            self.log.error('Try to load file at path %s which does not exist')
            return False

        self._scene_3d.scene.clear_geometry()
        
        mesh = io.read_triangle_mesh(filepath)
        self._scene_3d.scene.add_geometry('main_geometry', mesh, rendering.Material())

        return True

    def _clear_scene(self) -> None:
        pass

    def create_buttons(self) -> None:
        pass

    def create_main_grid(self) -> None:
        pass

    def create_3D_scene(self) -> None:
        self._scene_3d = gui.SceneWidget()
        self._scene_3d.scene = rendering.Open3DScene(self.window.renderer)
        self._scene_3d.scene.show_axes(True)

    def _open_loading_dialog(self) -> None:
        pass
