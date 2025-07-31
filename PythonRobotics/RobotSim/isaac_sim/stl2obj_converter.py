import trimesh
import os
import glob

file_dir = "/home/yxtang/CodeBase/PythonCourse/PythonRobotics/RobotSim/isaac_sim/assets/urdf/ur_e_description/meshes/ur5e/visual"
file_pattern = os.path.join(file_dir, "*.dae")


stl_files = glob.glob(file_pattern)

for stl_file in stl_files:
    print(stl_file)

    # load the STL file
    mesh = trimesh.load_mesh(stl_file)

    if hasattr(mesh, "visual") and hasattr(mesh.visual, "material"):
        print("Materials found:", mesh.visual.material)
    else:
        print("No materials found.")

    # export as OBJ file
    generate_file_name = stl_file.replace("dae", "obj")
    mesh.export(generate_file_name)
