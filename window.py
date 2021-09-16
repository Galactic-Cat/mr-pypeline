'''Main Window class'''

from logging import getLogger
from os.path import exists

from open3d import geometry, io
from open3d.visualization import gui, rendering

class MainWindow():
    '''Class for drawing the main window'''
    ACTION_LOAD_MESH = 1
    ACTION_CLEAR_MESH = 2
    log = getLogger('MainWindow')
    
    def __init__(self, width:int, height:int):
        '''Initializes a MainWindow instance with a predefined height and width
        
        Args:
            width (int): The width of the window
            height (int): The height of the window
        '''
        
        self.window = gui.Application.instance.create_window("MR-pypeline", width, height)
        
        #self.window
        self.create_menu_bar()
        self.create_3D_scene()
        
        #self.window.add_child(self._scene_3d)
        #self.window.add_child(self._options_panel)

    def create_menu_bar(self):
        if gui.Application.instance.menubar is None:
            action_menu = gui.Menu()
            action_menu.add_item("Load Mesh", MainWindow.ACTION_LOAD_MESH)
            action_menu.add_item("Clear Mesh", MainWindow.ACTION_CLEAR_MESH)
        
        main_menu = gui.Menu()
        main_menu.add_menu("Actions", action_menu)

        if main_menu is None:
            self.log.error("Main menu could not be instantiated")

        gui.Application.instance.menubar = main_menu

        self.window.set_on_menu_item_activated(MainWindow.ACTION_LOAD_MESH, self.on_load_mesh)
        self.window.set_on_menu_item_activated(MainWindow.ACTION_CLEAR_MESH, self.on_clear_scene)

    def load(self, filepath) -> None:

        if not exists(filepath):
            self.log.error('Try to load file at path %s which does not exist')
            return 
        
        self._scene_3d.scene.clear_geometry()

        #open3d.io.read_file_geometry_type maybe use this to figure out which one we have to use
        #if geometry type &  io.CONTAINS_TRIANGLES: else log trying to load an unsupported file.
        mesh = io.read_triangle_mesh(filepath)
        mesh.compute_vertex_normals()

        #Define Mesh Material
        material = rendering.Material()
        material.base_color = [1,0,0.5,1]
        material.shader = 'defaultLit'

        #Add model to the scene
        self._scene_3d.scene.add_geometry('main_geometry', mesh, material)


    def on_load_mesh(self) -> None:

        '''Attempts to load a file from a path into the viewport
        Args:
            filepath (str): The file path
        Returns:
            bool: Whether the loading succeeded
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
        bbox = geometry.AxisAlignedBoundingBox([-15, -15, -15],
                                                   [15, 15, 15])
        self._scene_3d.setup_camera(60, bbox, [0, 0, 0])
        self._scene_3d.scene.show_axes(True)
        self.window.add_child(self._scene_3d)