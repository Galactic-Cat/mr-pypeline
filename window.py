'''Main Window class'''

from open3d.visualization import gui, rendering

class MainWindow():
    '''Class for drawing the main window'''

    def __init__(self, width: int, height: int) -> None:
        '''Initializes a MainWindow instance with a predefined height and width

        Args:
            width (int): The width of the window
            height (int): The height of the window
        '''
        self.window = gui.Application.instance.create_window("MR-pypeline", width, height)
        
        self.create_3D_scene()


        f_size = self.window.theme.font_size
        element_sep = int(round(1.5 * f_size))

        # Create options pannel
        self._options_panel = gui.Vert(0, gui.Margins(0.5*f_size, 0.5*f_size, 0.5*f_size, 0.5*f_size))

        # Create buttons for loading pannel
        self._load_mesh_button = gui.Button("Add Mesh")
        self._load_mesh_button.horizontal_padding_em = 0.2
        self._load_mesh_button.vertical_padding_em = 0.1
        self._load_mesh_button.set_on_clicked(self._open_loading_dialog) #TODO
        
        self._clear_mesh_button = gui.Button("Clear Mesh")
        self._clear_mesh_button.horizontal_padding_em = 0.2
        self._clear_mesh_button.vertical_padding_em = 0.1
        self._clear_mesh_button.set_on_clicked(self._clear_scene) #TODO


        #Create layout inside options panel

        h_grid = gui.Horiz(0.25 * f_size)
        h_grid.add_stretch()
        h_grid.add_child(self._load_mesh_button)
        h_grid.add_child(self._clear_mesh_button)
        h_grid.add_stretch()
        self._options_panel.add_child(h_grid)

        self.window.add_child(self._Scene3D)
        self.window.add_child(self._options_panel)

    
    def load(self, filepath) -> bool:
    
        if not filepath:
            return False
        
        self._Scene3D.scene.clear_geometry()

        #geometry_type = 
        


    def _open_loading_dialog(self) -> None:
        pass

    def _clear_scene(self) -> None:
        pass

    def create_buttons(self) -> None:
        pass
    
    def create_main_grid(self) -> None:
        pass
    
    def create_3D_scene(self) -> None:
        self._Scene3D = gui.SceneWidget()
        self._Scene3D.scene = rendering.Open3DScene(self.window.renderer)
