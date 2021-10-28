'''Main Window class'''

from logging import getLogger
from os import getcwd, listdir
import os.path
from posixpath import basename
from numpy.lib.npyio import load

from open3d import geometry, io
from open3d.visualization import gui, rendering
from search import Search
from pandas import DataFrame

basedir = os.path.dirname(os.path.realpath(__file__))

class MainWindow():
    '''Class for drawing the main window'''
    
    # Menu statics
    ACTION_LOAD_MESH = 1
    ACTION_CLEAR_MESH = 2
    MENU_SHOW_COMPARE = 11

    SEARCH_SAMPLE = 8


    log = getLogger('MainWindow')

    def __init__(self, width:int, height:int):
        '''Initializes a MainWindow instance with a predefined height and width
        
        Args:
            width (int): The width of the window
            height (int): The height of the window
        '''

        self.window = gui.Application.instance.create_window("MR-pypeline", width, height)
        self.search_engine = Search('output\preprocess\database.csv')
        self.results = None

        self.create_3D_scene()
        self.create_search_panel()

        self.window.set_on_layout(self._on_layout)

        self.create_menu_bar()

    # Creation Functionality

    def create_search_panel(self):
        self.search_results = [] # Define it for later use.

        em = self.window.theme.font_size

        #Create Top level layout
        self._search_panel = gui.Vert(0, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em,
                                         0.5 * em))

        #Add "search bar" that opens a dialog box to load the mesh you want to find object similar to it.
        self._search_txtbox = gui.TextEdit()

        #Add button that will open the search
        findfilebutton = gui.Button("...")
        findfilebutton.horizontal_padding_em = 0.5
        findfilebutton.vertical_padding_em = 0
        findfilebutton.set_on_clicked(self.on_search_mesh)

        #Create horizontal layout for "search bar"
        search_layout = gui.Horiz()

        #add label that indicate current file
        search_layout.add_child(gui.Label("Current file"))

        #Add elements to the horizontal layout.
        search_layout.add_child(self._search_txtbox)
        search_layout.add_fixed(0.25 * em)
        search_layout.add_child(findfilebutton)
        
        self._search_panel.add_child(search_layout)
        title_layout = gui.Horiz()
        title_layout.add_stretch()
        title_layout.add_child(gui.Label("Query Results"))
        title_layout.add_stretch()

        self._list_widget = gui.ListView()
        self._list_widget.selected_index = -1
        self._list_widget.set_on_selection_changed(self._on_list)

        self._search_panel.add_child(self._search_txtbox)
        self._search_panel.add_child(title_layout)
        self._search_panel.add_child(self._list_widget)
        self.window.add_child(self._search_panel)

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        self._scene_3d.frame = r
        width = 17 * layout_context.theme.font_size
        height = min(
            r.height,
            self._search_panel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).height)
        self._search_panel.frame = gui.Rect(r.get_right() - width, r.y, width,
                                              height)

    def create_menu_bar(self):
        if gui.Application.instance.menubar is None:
            action_menu = gui.Menu()
            action_menu.add_item("Load Mesh", MainWindow.ACTION_LOAD_MESH)
            action_menu.add_item("Clear Mesh", MainWindow.ACTION_CLEAR_MESH)

            search_menu = gui.Menu()
            search_menu.add_item("Compare Mesh", MainWindow.MENU_SHOW_COMPARE)
            search_menu.set_checked(MainWindow.MENU_SHOW_COMPARE, True)

        main_menu = gui.Menu()
        main_menu.add_menu("Actions", action_menu)
        main_menu.add_menu("Search", search_menu)

        if main_menu is None:
            self.log.error("Main menu could not be instantiated")

        gui.Application.instance.menubar = main_menu

        self.window.set_on_menu_item_activated(MainWindow.ACTION_LOAD_MESH, self.on_load_mesh)
        self.window.set_on_menu_item_activated(MainWindow.ACTION_CLEAR_MESH, self.on_clear_scene)
        self.window.set_on_menu_item_activated(MainWindow.MENU_SHOW_COMPARE, self.on_menu_toggle_search)

    def create_file_dialog(self) -> gui.FileDialog:
        
        load_dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose a mesh to load",
                            self.window.theme)
        
        load_dlg.add_filter(".ply .off", "Mesh files (.ply .off)")
        load_dlg.add_filter(".off", "Object File Format (.off)")
        load_dlg.add_filter(".ply", "Polygon Files (.ply)")

        load_dlg.set_on_cancel(self.window.close_dialog)

        return load_dlg

    #Load functionality

    def load(self, path: str, use_wireframe: bool = True) -> None:
        '''Loads the file into a shape to render the mesh into the scene.
        
        Args:
            path (str): Path from which to load the mesh file.
        '''
        if not os.path.exists(path):
            self.log.error('Try to load file at path %s which does not exist', path)
            return 
        
        # Load and prepare the mesh and material
        mesh_material = rendering.Material()
        mesh_material.base_color = [1,0,0.5,1]
        mesh_material.shader = 'defaultLit'
        mesh = io.read_triangle_mesh(path)
        
        mesh.compute_vertex_normals()

        # Create the Wireframe, if necessary
        if use_wireframe:
            wireframe = geometry.LineSet.create_from_triangle_mesh(mesh)
            wireframe_material = rendering.Material()
            wireframe_material.base_color = [1,1,1,1]
            wireframe_material.shader = 'defaultLit'
        
        #Add models to the scene
        self._scene_3d.scene.clear_geometry()
        self._scene_3d.scene.add_geometry('main_geometry', mesh, mesh_material)
        
        if use_wireframe:
            self._scene_3d.scene.add_geometry('wireframe', wireframe, wireframe_material)

    def on_load_mesh(self) -> None:
        '''Attempts to load a file from a path into the viewport

        Args:
            filepath (str): The file path
        '''
        load_dlg = self.create_file_dialog()

        load_dlg.set_on_done(self._on_load_dialog_done)

        self.window.show_dialog(load_dlg)

    def _on_load_dialog_done(self, path: str) -> None:
        '''Closes the file loading dialog and loads the mesh

        Args:
            path (str): The selected path
        '''                
        self.window.close_dialog()
        self.load(path)
    
    #Search functionality

    def on_menu_toggle_search(self) -> None:
        self._search_panel.visible = not self._search_panel.visible
        gui.Application.instance.menubar.set_checked(
            MainWindow.MENU_SHOW_COMPARE, self._search_panel.visible)
    

    def display_search_results(self, results: DataFrame) -> None:
        self.results = results[['path']].head(self.SEARCH_SAMPLE)
        items = [basedir + entry[1:] for entry in self.results['path']]
        self._list_widget.set_items(items)
    
    def _on_list(self, new_val, is_dlb_click):
        self.load(new_val)

    def on_search_mesh(self):
        '''Attempts to load a file from a path into the viewport

        Args:
            filepath (str): The file path
        '''
        load_dlg = self.create_file_dialog()

        load_dlg.set_on_done(self._on_search_dialog_done)

        self.window.show_dialog(load_dlg)
    
    def search_similar_to(self, path: str):

        results = self.search_engine.compare(path)
        self.display_search_results(results)

    def _on_search_dialog_done(self, path:str) -> None:
        self.window.close_dialog()
        self._search_txtbox.text_value = path
        self.search_similar_to(path)
        self.load(path)
        self.window.close_dialog()


    #Clearing functionality

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
