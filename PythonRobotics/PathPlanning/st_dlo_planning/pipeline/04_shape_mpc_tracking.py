if __name__ == '__main__':
    import sys
    import os
    import pathlib
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from pprint import pprint
    from omegaconf import OmegaConf

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)
    
    from st_dlo_planning.neural_mpc_tracker.gdm_dataset import visualize_shape
    from st_dlo_planning.neural_mpc_tracker.policy import GradientLMPCAgent
    from st_dlo_planning.neural_mpc_tracker.modelling_gdm import GDM, GDM_CFG
    from st_dlo_planning.envs.planar_cable_deform.planar_cable_deform import DualGripperCableEnv, wrap_angle
    import st_dlo_planning.utils.pytorch_utils as ptu
    import st_dlo_planning.utils.misc_utils as misu
    import seaborn as sns
    from st_dlo_planning.utils.world_map import Block, WorldMap, MapCfg
    from st_dlo_planning.neural_mpc_tracker.mpc_solver import relaxed_log_barrier
    from omegaconf import OmegaConf
    
    import dill
    import time

    seed = 20
    misu.setup_seed(seed)
    device = ptu.init_gpu(use_gpu=False)

    Q = np.eye(6, 6) * 5.

    # load the planned dlo configuration sequence
    map_case = 'map_case8'
    cfg_path = f'/home/yxtang/CodeBase/PythonCourse/PythonRobotics/PathPlanning/st_dlo_planning/envs/map_cfg/{map_case}.yaml'
    map_cfg_file = OmegaConf.load(cfg_path)
    
    map_cfg = MapCfg(resolution=map_cfg_file.workspace.resolution,
                     map_xmin=map_cfg_file.workspace.map_xmin,
                     map_xmax=map_cfg_file.workspace.map_xmax,
                     map_ymin=map_cfg_file.workspace.map_ymin,
                     map_ymax=map_cfg_file.workspace.map_ymax,
                     map_zmin=map_cfg_file.workspace.map_zmin,
                     map_zmax=map_cfg_file.workspace.map_zmax,
                     robot_size=map_cfg_file.workspace.robot_size,
                     dim=3)
    
    world_map = WorldMap(map_cfg)

    # ============== add some obstacles =========================
    size_z = map_cfg_file.workspace.map_zmax
    obstacles = map_cfg_file.obstacle_info.obstacles
    i = 0
    for obstacle in obstacles:
        world_map.add_obstacle(Block(obstacle[0], obstacle[1], size_z, 
                                     obstacle[2], obstacle[3], angle=obstacle[4]*np.pi, clr=[0.3+0.01*i, 0.5, 0.4]))
        i += 1
    world_map.finalize()
    
    obs_info = world_map.get_obs_tensor_info(device)
    for ofo in obs_info:
        pprint(ofo)
        print(' +++++++ ' * 5)
    
    obs_vrtxs = world_map.get_obs_vertixs()

    result_path = pathlib.Path(__file__).parent.parent.joinpath('results', f'{map_case}_optimal_shape_seq.npy')
    planned_shape_seq = np.load(result_path, mmap_mode='r')
    planned_shape_seq = np.copy( planned_shape_seq.reshape(-1, 13, 3) )

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(projection='3d')
    for p in range(0, planned_shape_seq.shape[0], 2):
        visualize_shape(planned_shape_seq[p], ax, ld=1, clr='b')
    plt.axis('equal')
    plt.show()

    _, ax = world_map.visualize_passage(full_passage=False)

    for obs_vrtx in obs_vrtxs:
        print(obs_vrtx)
        for s in obs_vrtx:
            ax.scatter(s[0], s[1], c='r')
    plt.axis('equal')
    plt.show()

    # ================ config the mpc solver ===================
    def angle_with_x_axis(p1, p2):
        delta_x = p2[0] - p1[0]
        delta_y = p2[1] - p1[1]
        angle = torch.atan2(delta_y, delta_x)
        return angle

    # @torch.jit.script
    def path_cost_func_tensor(dlo_kp_2d:torch.Tensor, eef_states:torch.Tensor, 
                              u:torch.Tensor, target_dlo_kp_2d:torch.Tensor, phi):
        device = torch.device('cpu')

        # dlo
        # seprate the shape into position and shape parts
        weight_position = torch.diag(torch.tensor([1.0, 1.0], device=device)) * 150.0
        weight_curve = torch.diag(torch.tensor([0.2, 1.0] + [0.2, 1.0] * (13-2) + [0.2, 1.0], device=device)) * 150.0
        
        target_position = torch.mean(target_dlo_kp_2d, dim=0, keepdim=True)
        dlo_position = torch.mean(dlo_kp_2d, dim=0, keepdim=True)
        
        target_curve = target_dlo_kp_2d - target_position
        dlo_curve = dlo_kp_2d - dlo_position

        pos_err = target_position - dlo_position
        curve_err = target_curve - dlo_curve

        curve_err = curve_err.flatten()
        pos_err = pos_err.flatten()

        position_loss = 0.5 * (pos_err.T @ weight_position @ pos_err)
        curve_loss = 0.5 * (curve_err.T @ weight_curve @ curve_err)

        # obs_info
        obs_loss = 0.
        ofo = obs_info[0]
        for ofo in obs_info:
            vertexs = ofo['vertex']
            norm_vecs = ofo['normal_vec']
            for i in range(dlo_kp_2d.shape[0]):
                dlo_kp = dlo_kp_2d[i]
                sdf1 = torch.dot(dlo_kp - vertexs[0], norm_vecs[0] / torch.norm(norm_vecs[0]))
                sdf2 = torch.dot(dlo_kp - vertexs[1], norm_vecs[1] / torch.norm(norm_vecs[1]))
                sdf3 = torch.dot(dlo_kp - vertexs[2], norm_vecs[2] / torch.norm(norm_vecs[2]))
                sdf4 = torch.dot(dlo_kp - vertexs[3], norm_vecs[3] / torch.norm(norm_vecs[3]))

                # x = torch.tensor([sdf1, sdf2, sdf3, sdf4], device=device)
                # log_tmp = torch.log( torch.sum( torch.exp( 200. * x ) ) )
                # sdf_val = 1. / 200. * log_tmp

                sdf_val = torch.max(torch.tensor([sdf1, sdf2, sdf3, sdf4]))

                # if sdf_val < 0.05:
                obs_loss = obs_loss + relaxed_log_barrier(sdf_val, phi=phi)

            for m in range(2):
                eef_pos = eef_states[m, 0:2]

                sdf1 = torch.dot(eef_pos - vertexs[0], norm_vecs[0] / torch.norm(norm_vecs[0]))
                sdf2 = torch.dot(eef_pos - vertexs[1], norm_vecs[1] / torch.norm(norm_vecs[1]))
                sdf3 = torch.dot(eef_pos - vertexs[2], norm_vecs[2] / torch.norm(norm_vecs[2]))
                sdf4 = torch.dot(eef_pos - vertexs[3], norm_vecs[3] / torch.norm(norm_vecs[3]))

                # x = torch.tensor([sdf1, sdf2, sdf3, sdf4], device=device)
                # log_tmp = torch.log( torch.sum( torch.exp( 200. * x ) ) )
                # sdf_val = 1. / 200. * log_tmp

                sdf_val = torch.max(torch.tensor([sdf1, sdf2, sdf3, sdf4]))

                # if sdf_val < 0.05:
                obs_loss = obs_loss + relaxed_log_barrier(sdf_val, phi=phi)

        # eef_orientation
        left_theta = eef_states[0, 2]
        right_theta = eef_states[1, 2]
        left_theta_target = wrap_angle(angle_with_x_axis(target_dlo_kp_2d[1], target_dlo_kp_2d[0]))
        right_theta_target = wrap_angle(angle_with_x_axis(target_dlo_kp_2d[-1], target_dlo_kp_2d[-2]))
        # print(left_theta_target.data.item(), right_theta_target.data.item())

        # ctrl
        weight_u = torch.diag(torch.tensor([1.0, 1.0, 1.5] * 2, device=device)) * 5
        ctrl = u.flatten()
        
        ctrl_loss = 0.5 * (ctrl.T @ weight_u @ ctrl)

        orien_loss = (left_theta_target - left_theta) ** 2 + (right_theta_target - right_theta) ** 2
        c = position_loss + curve_loss + ctrl_loss + 0 * obs_loss + orien_loss * 0.01
        return c 
    
    # @torch.jit.script
    def final_cost_func_tensor(dlo_kp_2d:torch.Tensor, eef_states:torch.Tensor, target_dlo_kp_2d:torch.Tensor,
                               phi):
        device = torch.device('cpu')

        target_position = torch.mean(target_dlo_kp_2d, dim=0, keepdim=True)
        dlo_position = torch.mean(dlo_kp_2d, dim=0, keepdim=True)

        target_curve = target_dlo_kp_2d - target_position
        dlo_curve = dlo_kp_2d - dlo_position

        pos_err = target_position - dlo_position
        curve_err = target_curve - dlo_curve

        curve_err = curve_err.flatten()
        pos_err = pos_err.flatten()

        weight_position = torch.diag(torch.tensor([1.0, 1.0], device=device)) * 150.0
        weight_curve = torch.diag(torch.tensor([0.2, 1.0] + [0.2, 1.0] * (13-2) + [0.2, 1.0], device=device)) * 150.0

        position_loss = 0.5 * (pos_err.T @ weight_position @ pos_err)
        curve_loss = 0.5 * (curve_err.T @ weight_curve @ curve_err)

        # eef_orientation
        left_theta = eef_states[0, 2]
        right_theta = eef_states[1, 2]

        left_theta_target = wrap_angle(angle_with_x_axis(target_dlo_kp_2d[1], target_dlo_kp_2d[0]))
        right_theta_target = wrap_angle(angle_with_x_axis(target_dlo_kp_2d[-1], target_dlo_kp_2d[-2]))

        # print(left_theta.data.item(), right_theta.data.item())
        # print(left_theta_target.data.item(), right_theta_target.data.item())

        # obs_info
        obs_loss = 0.
        ofo = obs_info[0]
        for ofo in obs_info:
            vertexs = ofo['vertex']
            norm_vecs = ofo['normal_vec']
            for i in range(dlo_kp_2d.shape[0]):
                dlo_kp = dlo_kp_2d[i]
                sdf1 = torch.dot(dlo_kp - vertexs[0], norm_vecs[0] / torch.norm(norm_vecs[0]))
                sdf2 = torch.dot(dlo_kp - vertexs[1], norm_vecs[1] / torch.norm(norm_vecs[1]))
                sdf3 = torch.dot(dlo_kp - vertexs[2], norm_vecs[2] / torch.norm(norm_vecs[2]))
                sdf4 = torch.dot(dlo_kp - vertexs[3], norm_vecs[3] / torch.norm(norm_vecs[3]))

                # x = torch.tensor([sdf1, sdf2, sdf3, sdf4], device=device)
                # log_tmp = torch.log( torch.sum( torch.exp( 200. * x ) ) )
                # sdf_val = 1. / 200. * log_tmp

                sdf_val = torch.max(torch.tensor([sdf1, sdf2, sdf3, sdf4]))

                # if sdf_val < 0.05:
                obs_loss = obs_loss + relaxed_log_barrier(sdf_val, phi=phi) #1 / (sdf_val**4)

            for m in range(2):
                eef_pos = eef_states[m, 0:2]
                sdf1 = torch.dot(eef_pos - vertexs[0], norm_vecs[0] / torch.norm(norm_vecs[0]))
                sdf2 = torch.dot(eef_pos - vertexs[1], norm_vecs[1] / torch.norm(norm_vecs[1]))
                sdf3 = torch.dot(eef_pos - vertexs[2], norm_vecs[2] / torch.norm(norm_vecs[2]))
                sdf4 = torch.dot(eef_pos - vertexs[3], norm_vecs[3] / torch.norm(norm_vecs[3]))

                # x = torch.tensor([sdf1, sdf2, sdf3, sdf4], device=device)
                # log_tmp = torch.log( torch.sum( torch.exp( 200. * x ) ) )
                # sdf_val = 1. / 200. * log_tmp

                sdf_val = torch.max(torch.tensor([sdf1, sdf2, sdf3, sdf4]))

                # if sdf_val < 0.05:
                obs_loss = obs_loss + relaxed_log_barrier(sdf_val, phi=phi) #1 / (sdf_val**4)
        
        orien_loss = (left_theta_target - left_theta) ** 2 + (right_theta_target - right_theta) ** 2
        
        c = position_loss + curve_loss + orien_loss * 0.01 + 0 * obs_loss
        return c


    # setup the simulation env
    mode = 'human' #'rgb_array' #'depth_array' #
    camera = 'topview'
    task_id = '03'
    env = DualGripperCableEnv(task=task_id, feat_stride=3, render_mode=mode, camera_name=camera)
    obs, _ = env.reset()

    # env.step_relative_eef([0.05, 0.05, -0.01], [0.05, 0.05, 0.01], num_steps=100)

    # ====================== global deformation model ===================
    model_dir = '/home/yxtang/CodeBase/PythonCourse/PythonRobotics/PathPlanning/st_dlo_planning/results/checkpoints/st_dlo_gdm_30/'
    model_ckpt_path = model_dir + 'latest.ckpt'

    gdm_model = GDM(GDM_CFG())

    model_params = torch.load(model_ckpt_path, pickle_module=dill)['state_dicts']['model']
    gdm_model.load_state_dict(model_params)
    gdm_model.to(device)

    lb = np.array([-0.04, -0.04, -0.4, -0.04, -0.04, -0.4]) / 5
    ub = np.array([ 0.04,  0.04,  0.4,  0.04,  0.04,  0.4]) / 5

    H = 2
    agent = GradientLMPCAgent(nx=env.num_feat*2, 
                              nu=env.num_grasp*3,
                              dlo_length=env.dlo_len, 
                              dt=env.dt,
                              path_cost_func=path_cost_func_tensor, 
                              final_cost_func=final_cost_func_tensor,
                              obstacle_vertixs=obs_vrtxs,
                              Q=Q,
                              device=device,
                              umax=ub,
                              umin=lb,
                              discount_factor=0.8, pretrained_model=gdm_model,
                              traj_horizon=H, log_prefix='', mdl_ft=True) #False

    T = 7000
    error_list = []
    ultra_error_list = []
    action_list = []
    cons_vio_list = []

    dlos = []
    target_dlos = []

    error_pre = 1e3
    obj_loss_pre = 1e7
    pre_action = 0.
    
    i = 0

    ref_idx = 0
    patience = 0
    final_target_shape = planned_shape_seq[-1].flatten()

    artists = []
    ani_dlos = []

    while i < T:
        tc = time.time()

        dlo_kp = obs['dlo_keypoints']
        eef_states = obs['lowdim_eef_transforms']
        dlo_kp_ref = planned_shape_seq[ref_idx]

        if ref_idx == 0:
            delta_eef, obj_loss, hmin = agent.select_action(dlo_kp, eef_states, dlo_kp_ref, safe_mode=False)
        else:
            delta_eef, obj_loss, hmin = agent.select_action(dlo_kp, eef_states, dlo_kp_ref, safe_mode=False)

        action = delta_eef

        action = np.clip(action, lb, ub)

        # action = 0.5 * pre_action + 0.5 * action

        next_obs, reward, done, truncated, info = env.step_relative_eef(action[0:3], action[3:], num_steps=1, render_mode='human')
        # env.render() 
        #  
        # ================== model adaptation =================== #
        # Local Jacobian update
        agent.update_local_Jacobian(obs, action, next_obs)
        # ============================================================ #
        
        obs = next_obs

        dlo_kp = obs['dlo_keypoints']

        dlo_kp_3d = dlo_kp.reshape(-1, 3)
        tmp = []
        for j in range(env.num_feat):
            for obstacle in world_map._obstacle:
                cle, _ = obstacle.get_2d_sdf_val([dlo_kp_3d[j][0], dlo_kp_3d[j][1]])
                tmp.append(cle)
        
        cons_vio = np.min(np.array(tmp))

        error = np.linalg.norm(dlo_kp - dlo_kp_ref.flatten(), 2) / env.num_feat

        ultra_error = np.linalg.norm(dlo_kp - final_target_shape, 2 ) / env.num_feat
        error_list.append(error)
        ultra_error_list.append(ultra_error)
        action_list.append(action)
        cons_vio_list.append(cons_vio)

        if i % 5 == 0:
            print(f'env_step: {i}, ref_idx {ref_idx}/{planned_shape_seq.shape[0]}, inter_err: {error}, ultra_err: {ultra_error}, hmin: {cons_vio}')

        i += 1
        patience += 1
        pre_action = action

        if ref_idx > 0:
            ani_dlos.append(dlo_kp.reshape(-1, 3)[:, 0:2])

        if error < 5e-3 or (patience > 200 and ref_idx > 0):
            ref_idx += 1
            ref_idx = min(ref_idx, planned_shape_seq.shape[0] - 1)
            patience = 0

            dlos.append(dlo_kp.reshape(-1, 3))
            target_dlos.append(dlo_kp_ref.reshape(-1, 3))
            

        if ultra_error <= 3e-3:
            dlos.append(dlo_kp.reshape(-1, 3))
            target_dlos.append(final_target_shape.reshape(-1, 3))
            break

    # ======================= plot ======================== #
    import matplotlib.pyplot as plt
    import seaborn as sns
    from st_dlo_planning.utils.path_interpolation import visualize_shape
    
    sns.set_theme('notebook')
    plt.figure(1)
    sns.lineplot(error_list)
    sns.lineplot(ultra_error_list)

    plt.figure(2)
    for j in range(6):
        sns.lineplot([action[j] for action in  action_list])
    plt.show()

    plt.figure(3)
    sns.lineplot(cons_vio_list)
    plt.show()
    
    _, ax = world_map.visualize_passage(full_passage=False)

    for k in range(0, len(dlos), 1):
        dlo_shape = dlos[k]
        dlo_shape = dlo_shape.reshape(-1, 3)
        visualize_shape(dlo_shape, ax, clr='k', ld=2.0)
        visualize_shape(target_dlos[k], ax, clr='r', ld=1.0)
    plt.axis('equal')
    plt.savefig(f"/home/yxtang/CodeBase/PythonCourse/PythonRobotics/PathPlanning/st_dlo_planning/results/tracking_res_{map_case}.png",
                dpi=2000)
    plt.show()

    fig, ax = world_map.visualize_passage(full_passage=False)
    clrs = np.linspace(0.0, 1.0, len(ani_dlos))
    rever_clrs = np.flip(clrs)
    import matplotlib.animation as animation
    for i in range(0, len(ani_dlos), 2):
        container1 = ax.plot(ani_dlos[i][:, 0], ani_dlos[i][:, 1], color=[clrs[i], rever_clrs[i], clrs[i]], linewidth=3)
        artists.append(container1)
        print(f'Frame: {i}')
    plt.axis('equal')
    ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=20)
    ani.save(filename=f"/home/yxtang/CodeBase/PythonCourse/PythonRobotics/PathPlanning/st_dlo_planning/results/tracking_{map_case}.gif", writer="pillow")