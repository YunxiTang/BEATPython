import viser
import numpy as np
from typing import TypedDict
import numpy.typing as npt
import tyro

from plyfile import PlyData
from viser import transforms as tf
import time
from pathlib import Path


class SplatFile(TypedDict):


    centers: npt.NDArray[np.floating]

    rgbs: npt.NDArray[np.floating]

    opacities: npt.NDArray[np.floating]

    covariances: npt.NDArray[np.floating]


def load_ply_file(ply_file_path: Path, center: bool = False) -> SplatFile:

    start_time = time.time()


    SH_C0 = 0.28209479177387814


    plydata = PlyData.read(ply_file_path)

    v = plydata["vertex"]

    positions = np.stack([v["x"], v["y"], v["z"]], axis=-1)

    scales = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1)) / 100

    wxyzs = np.stack([v["rot_w"], v["rot_x"], v["rot_y"], v["rot_z"]], axis=1)

    colors = 0.5 + SH_C0 * np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1)

    opacities = 1.0 / (1.0 + np.exp(-v["opacity"][:, None]))


    Rs = tf.SO3(wxyzs).as_matrix()

    covariances = np.einsum(

        "nij,njk,nlk->nil", Rs, np.eye(3)[None, :, :] * scales[:, None, :] ** 2, Rs

    )

    if center:

        positions -= np.mean(positions, axis=0, keepdims=True)


    num_gaussians = len(v)

    print(

        f"PLY file with {num_gaussians=} loaded in {time.time() - start_time} seconds"

    )

    return {

        "centers": positions,

        "rgbs": colors,

        "opacities": opacities,

        "covariances": covariances,

    }



def main(

    splat_paths: tuple[Path, ...] = (

        # Path(__file__).absolute().parent.parent / "assets" / "train.splat",

        Path(__file__).absolute().parent.parent / "assets" / "nike.splat",

    ),

) -> None:

    server = viser.ViserServer()
    splat_paths = ['/home/tyx/CodeBase/BEATPython/PythonRobotics/RoboticVision/GaussianSplat/output/point_cloud/iteration_19999/point_cloud.ply',]

    server.scene.add_grid(
        name='ground',
        width=10,
        height=10.,
        plane='xy',
        cell_color=[125,125,125]
    )

    for i, splat_path in enumerate(splat_paths):
        splat_path = Path(splat_path)

        if splat_path.suffix == ".ply":

            splat_data = load_ply_file(splat_path, center=True)

        else:

            raise SystemExit("Please provide a filepath to a .splat or .ply file.")


        server.scene.add_transform_controls(f"/{i}")

        gs_handle = server.scene.add_gaussian_splats(

            f"/{i}/gaussian_splats",

            centers=splat_data["centers"],

            rgbs=splat_data["rgbs"],

            opacities=splat_data["opacities"],

            covariances=splat_data["covariances"],

        )


        remove_button = server.gui.add_button(f"Remove splat object {i}")


        @remove_button.on_click

        def _(_, gs_handle=gs_handle, remove_button=remove_button) -> None:

            gs_handle.remove()

            remove_button.remove()


    while True:

        time.sleep(10.0)



if __name__ == "__main__":

    tyro.cli(main)