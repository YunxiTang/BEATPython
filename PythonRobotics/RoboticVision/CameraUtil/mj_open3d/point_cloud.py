import mujoco
# import open3d


if __name__ == '__main__':
    xml = """
    <mujoco>
    <worldbody>
        <light name="top" pos="0 0 1"/>
        <body name="box_and_sphere" euler="0 0 -30">
        <joint name="swing" type="hinge" axis="1 -1 0" pos="-.2 -.2 -.2"/>
        <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
        <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
        </body>
    </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    renderer = mujoco.Renderer(model)
    data = mujoco.MjData(model)

    mujoco.mj_forward(model, data)
    renderer.update_scene(data)