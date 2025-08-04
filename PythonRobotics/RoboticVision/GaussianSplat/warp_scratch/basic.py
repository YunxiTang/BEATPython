import numpy as np
import torch
import warp as wp
tensor_type = torch.float32

if __name__ == "__main__":
    gaussian_centers_np = np.array([[2, 0, -2], [0, 2, -2], [-2, 0, -2]])
    gaussian_centers = torch.tensor(gaussian_centers_np, dtype=tensor_type)
    print("Gaussian centers:\n", gaussian_centers)
    
    gaussian_centers = wp.from_numpy(gaussian_centers_np)
    print("Gaussian centers (Warp tensor):\n", gaussian_centers)
    