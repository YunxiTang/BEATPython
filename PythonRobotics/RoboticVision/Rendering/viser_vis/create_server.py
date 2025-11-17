import viser
import time


if __name__ == '__main__':
    # create a server
    viser_server = viser.ViserServer()
    
    # add 3d objects to the scene
    box = viser_server.scene.add_box(
        name='/box',
        color=(12, 134, 124),
        dimensions=(1.0,1.0,1.0)
    )
    
    # add gui control
    box_vis = viser_server.gui.add_checkbox(
        label='box_vis',
        initial_value=True
    )
    box_color = viser_server.gui.add_rgb(
        label='box_color',
        initial_value=(12, 134, 124)
    )
    
    # Connect GUI controls to scene objects
    @box_vis.on_update
    def _(_):
        box.visible = box_vis.value
        
    @box_color.on_update
    def _(_):
        box.color = box_color.value
    
    print("Open your browser to http://localhost:8080")

    print("Press Ctrl+C to exit")


    while True:

        time.sleep(10.0)
    
    