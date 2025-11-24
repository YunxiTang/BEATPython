import random
import time

import viser

server = viser.ViserServer()

while True:
    # Add some coordinate frames to the scene. These will be visualized in the viewer.
    server.scene.add_frame(
        "/tree",
        wxyz=(1.0, 0.0, 0.0, 0.0),
        position=(2.0, 2.0, 0.2),
    )
    tree = server.scene.add_frame(
        "/tree/branch",
        wxyz=(1.0, 0.0, 0.0, 0.0),
        position=(random.random() * 2.0, 2.0, 0.2),
    )
    leaf = server.scene.add_frame(
        "/tree/branch/leaf",
        wxyz=(1.0, 0.0, 0.0, 0.0),
        position=(random.random() * 2.0, 2.0, 0.2),
    )

    # Move the leaf randomly. Assigned properties are automatically updated in
    # the visualizer.
    for i in range(10):
        tree.position = (random.random() * 2.0, 2.0, 0.2)
        time.sleep(0.1)

    time.sleep(0.1)