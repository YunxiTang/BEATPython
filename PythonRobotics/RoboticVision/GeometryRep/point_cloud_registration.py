from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from pydrake.all import (
    CsdpSolver,
    MathematicalProgram,
    PointCloud,
    RigidTransform,
    RotationMatrix,
    Solve,
)
from scipy.spatial import KDTree



def MakeRandomObjectModelAndScenePoints(num_model_points=20,
                                        noise_std=0,
                                        num_outliers=0,
                                        yaw_O=None,
                                        p_O=None,
                                        num_viewable_points=None,
                                        seed=None,
                                        ):
    """
        Returns p_Om, p_s
    """
    rng = np.random.default_rng(seed)

    # Make a random set of points to define an object in the x-y plane
    theta = np.arange(0, 2.0 * np.pi, 2.0 * np.pi / num_model_points)
    l = (
        1.0 + 0.5 * np.sin(2.0 * theta) + 0.4 * rng.random((1, num_model_points))
    )
    p_Om = np.vstack((l * np.sin(theta), l * np.cos(theta), 0 * l))

    # Make a random object pose if one is not specified, and apply it to get the scene points.
    if p_O is None:
        p_O = [2.0 * rng.random(), 2.0 * rng.random(), 0.0]

    if len(p_O) == 2:
        p_O.append(0.0)

    if yaw_O is None:
        yaw_O = 0.5 * rng.random()

    X_O = RigidTransform(RotationMatrix.MakeZRotation(yaw_O), p_O)

    if num_viewable_points is None:
        num_viewable_points = num_model_points

    assert num_viewable_points <= num_model_points
    p_s = X_O.multiply(p_Om[:, :num_viewable_points])
    p_s[:2, :] += rng.normal(scale=noise_std, size=(2, num_viewable_points))
    
    if num_outliers:
        outliers = rng.uniform(low=-1.5, high=3.5, size=(3, num_outliers))
        outliers[2, :] = 0
        p_s = np.hstack((p_s, outliers))

    return p_Om, p_s, X_O

