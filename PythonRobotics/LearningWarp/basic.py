import warp as wp
import numpy as np
import torch
import time

wp.init()

wp.config.quiet = True

if __name__ == '__main__':
    x_np = np.array([1.,2.,3.])
    print(x_np)
    x_wp = wp.from_numpy(x_np)
    print(x_wp)
    
    pos = wp.vec3(1., 1., 1.)
    print(pos)
    rot = wp.from_numpy(
        np.eye(3)
    )
    print(rot.shape, '\n', rot)
    
    y = wp.array(
        [[1.,2.,3.,4.],
         [1.,2.,3.,4.]], dtype=wp.float32
    )
    print(y, y.shape)
    
    @wp.struct
    class Gaussian:
        pos: wp.vec3
        rot: wp.quat
        scale: wp.vec3
        rgb: wp.vec3ui
        opacity: float
        
    g1 = Gaussian()
    g2 = Gaussian()
    g1.pos = wp.vec3(1,2,2)
    
    gaussian_arr = wp.array([g1, g2], dtype=Gaussian)
    print(gaussian_arr.dtype, gaussian_arr.device, gaussian_arr.shape)
    
    # ====== kernel ======
    print('======== kernel =========')
    @wp.kernel
    def avg_kernel(a:wp.array(dtype=float), b:wp.array(dtype=float), c:wp.array(dtype=float)):
        pid = wp.tid()
        c[pid] = (a[pid] + b[pid]) * 0.5
        
    x = wp.from_numpy(
        np.random.normal(0., 1.0, size=(1000,)),
        dtype=float
    )
    y = wp.array(
        torch.randn([1000,], dtype=torch.float),
        dtype=float
    )
    z = wp.zeros(1000, dtype=float)
    wp.launch(avg_kernel, dim=x.shape[0], inputs=[x, y], outputs=[z])
    print(z[0:10])
    
    @wp.kernel
    def range_fill_kernel(x: wp.array(dtype=int)):
        i = wp.tid()
        x[i] = i
    
    current_device = wp.get_device()
    print(current_device)
    out = wp.zeros([2000,], dtype=int)
    ts = time.time()
    for i in range(10):
        wp.launch(range_fill_kernel, dim=2000, inputs=[], outputs=[out])
    te = time.time() - ts
    print(te)