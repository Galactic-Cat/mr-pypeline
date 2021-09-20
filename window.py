'''Main Window class'''

from logging import getLogger
from os import getcwd, listdir
from os.path import exists, isdir, isfile, join, basename
from numpy.lib.npyio import load

from open3d import geometry, io
from open3d.visualization import gui, rendering
from shape import Shape


class MainWindow():
    '''Class for drawing the main window'''
    
    # Menu statics
    ACTION_LOAD_MESH = 1
    ACTION_CLEAR_MESH = 2
    READ_DATA_BASE = 3

    # Database paths
    PRINCETON_PATH = "./princeton"
    PSB_PATH = "./psb"

    log = getLogger('MainWindow')

    def __init__(self, width:int, height:int):
        '''Initializes a MainWindow instance with a predefined height and width
        
        Args:
            width (int): The width of the window
            height (int): The height of the window
        '''
        
        self.window = gui.Application.instance.create_window("MR-pypeline", width, height)
        
        self.shape: Shape = None
        self.create_menu_bar()
        self.create_3D_scene()
        

    def create_menu_bar(self):
        if gui.Application.instance.menubar is None:
            action_menu = gui.Menu()
            action_menu.add_item("Load Mesh", MainWindow.ACTION_LOAD_MESH)
            action_menu.add_item("Clear Mesh", MainWindow.ACTION_CLEAR_MESH)

            debug_menu = gui.Menu()
            debug_menu.add_item("Load Shape", MainWindow.READ_DATA_BASE)

        main_menu = gui.Menu()
        main_menu.add_menu("Actions", action_menu)
        main_menu.add_menu("Debug", debug_menu)

        if main_menu is None:
            self.log.error("Main menu could not be instantiated")

        gui.Application.instance.menubar = main_menu

        self.window.set_on_menu_item_activated(MainWindow.ACTION_LOAD_MESH, self.on_load_mesh)
        self.window.set_on_menu_item_activated(MainWindow.ACTION_CLEAR_MESH, self.on_clear_scene)
        self.window.set_on_menu_item_activated(MainWindow.READ_DATA_BASE, self.on_load_database_prince)

    def load(self, filepath) -> None:

        if not exists(filepath):
            self.log.error('Try to load file at path %s which does not exist')
            return 
        
        self._scene_3d.scene.clear_geometry()

        mesh = io.read_triangle_mesh(filepath)
        mesh.compute_vertex_normals()

        #Define Mesh Material
        material = rendering.Material()
        material.base_color = [1,0,0.5,1]
        material.shader = 'defaultLit'

        #Add model to the scene
        self._scene_3d.scene.add_geometry('main_geometry', mesh, material)


    def navigate_directory(self, path: str) -> None:
        
        if exists(path):
            for entry in listdir(path):
                f = join(path, entry)
                if isdir(f):
                    self.navigate_directory(f)
                if isfile(f):
                    file_ext = basename(f)[-4:] 
                    if  file_ext == '.off' or file_ext == '.ply':
                        self.log.debug("Reading file %s.", entry)
                        self.shape = Shape(f)
                        self.log.debug("Shape class initialized.")
                        #Todo calculate all shape properties and then update database! Rn we are just updating with Nulls causing a crash.
                        self.shape.update_database()
            return

        self.log.error("Navigating to %s failed.", path)
        return

    def on_load_database_prince(self) -> None:
        
        ''' Attemps to load database
        
        '''
        if not exists(self.PRINCETON_PATH):
            self.log.error("Princeton database path %s does not exists.", self.PRINCETON_PATH)

        princeton_files = getcwd() + self.PRINCETON_PATH + "\db"
        self.navigate_directory(princeton_files)

        return

    def on_load_database_psb(self) -> None:
        
        ''' Attemps to load PSB database
        
        '''
        if not exists(self.PRINCETON_PATH):
            self.log.error("Princeton database path %s does not exists.", self.PRINCETON_PATH)

        PSB_files = getcwd() + self.PSB_PATH
        self.navigate_directory(PSB_files)

        return

    def on_load_mesh(self) -> None:

        '''Attempts to load a file from a path into the viewport
        Args:
            filepath (str): The file path
        Returns:
            None
        '''

        load_dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose a mesh to load",
                            self.window.theme)
        load_dlg.add_filter(".ply .off", "Mesh files (.ply .off)")
        load_dlg.add_filter(".off", "Object File Format (.off)")
        load_dlg.add_filter(".ply", "Polygon Files (.ply)")

        load_dlg.set_on_cancel(self._on_load_dialog_cancel)
        load_dlg.set_on_done(self._on_load_dialog_done)

        self.window.show_dialog(load_dlg)

    def _on_load_dialog_done(self, filename):
        self.window.close_dialog()
        self.load(filename)


    def _on_load_dialog_cancel(self):
        self.window.close_dialog()

    def on_clear_scene(self):
        font_size = self.window.theme.font_size

        dialog = gui.Dialog("Clear")

        dlg_layout = gui.Vert(font_size, gui.Margins(font_size,font_size,font_size,font_size))
        dlg_layout.add_child(gui.Label("Are you sure you wish to clear the mesh?"))
        
        yes_button = gui.Button("Yes")
        yes_button.set_on_clicked(self._on_click_yes_btn)

        cancel_button = gui.Button("Cancel")
        cancel_button.set_on_clicked(self._on_click_cancel_btn)

        self.log.debug("Dialog buttons have been instantiated.")

        h_grid = gui.Horiz()
        h_grid.add_stretch()
        h_grid.add_child(yes_button)
        h_grid.add_stretch()
        h_grid.add_child(cancel_button)
        h_grid.add_stretch()
        dlg_layout.add_child(h_grid)
        dialog.add_child(dlg_layout)
    
        self.window.show_dialog(dialog)
    
    def _on_click_yes_btn(self):
        self._scene_3d.scene.clear_geometry()
        self.window.close_dialog()

    def _on_click_cancel_btn(self):
        self.window.close_dialog()

    def create_3D_scene(self) -> None:
        '''Creates the 3D Widget which will be used to render the meshes on the window'''
        #Instantiate scene
        self._scene_3d = gui.SceneWidget()
        self._scene_3d.scene = rendering.Open3DScene(self.window.renderer)

        #Set up scene settings
        self._scene_3d.scene.set_background([1, 1, 1, 1])
        self._scene_3d.scene.scene.set_sun_light(
            [-1, -1, -1],
            [1, 1, 1],
            1000)
        self._scene_3d.scene.scene.enable_sun_light(True)

        #Set up bounding box and camera
        bbox = geometry.AxisAlignedBoundingBox([-5, -5, -5],
                                                   [5, 5, 5])
        self._scene_3d.setup_camera(60, bbox, [0, 0, 0])
        self._scene_3d.scene.show_axes(True)
        self.window.add_child(self._scene_3d)
