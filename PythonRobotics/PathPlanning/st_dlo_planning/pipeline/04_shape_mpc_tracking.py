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
    from st_dlo_planning.temporal_config_opt.qp_solver import polish_dlo_shape
    import st_dlo_planning.utils.pytorch_utils as ptu
    import st_dlo_planning.utils.misc_utils as misu
    import seaborn as sns
    from st_dlo_planning.utils.world_map import Block, WorldMap, MapCfg, load_mapCfg_to_mjcf
    from st_dlo_planning.neural_mpc_tracker.mpc_solver import relaxed_log_barrier
    from st_dlo_planning.neural_mpc_tracker.gdm_dataset import ReplayBuffer
    from omegaconf import OmegaConf
    
    import dill
    import time

    def check_shape_validality(dlo_kp_3d, world_map):
        for j in range(len(dlo_kp_3d)):
            for obstacle in world_map._obstacle:
                cle, _ = obstacle.get_2d_sdf_val([dlo_kp_3d[j][0], dlo_kp_3d[j][1]])
                if cle < 0.0:
                    return False
        return True

    sns.set_theme('paper')

    seed = 20
    misu.setup_seed(seed)
    device = ptu.init_gpu(use_gpu=False)

    Q = np.eye(6, 6) * 50
    
    save = False
    # load the planned dlo configuration sequence
    map_case = 'camera_ready_so1'
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
    clrs = sns.color_palette("tab10", n_colors=max(3, len(obstacles))).as_hex()
    for obstacle in obstacles:
        world_map.add_obstacle(Block(obstacle[0], obstacle[1], size_z, 
                                     obstacle[2], obstacle[3], angle=obstacle[4]*np.pi, clr=clrs[i]))
        i += 1
    world_map.finalize()
    
    obs_info = world_map.get_obs_tensor_info(device)
    for ofo in obs_info:
        pprint(ofo)
        print(' +++++++ ' * 5)
    
    obs_vrtxs = world_map.get_obs_vertixs()

    result_path = pathlib.Path(__file__).parent.parent.joinpath('results', 'optimal_shape_seq_res', f'{map_case}_optimal_shape_seq.npy')
    planned_shape_seq = np.load(result_path, mmap_mode='r')
    planned_shape_seq = np.copy( planned_shape_seq.reshape(-1, 13, 3) )

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(projection='3d')
    for p in range(0, planned_shape_seq.shape[0], 2):
        visualize_shape(planned_shape_seq[p], ax, ld=1, clr='b')
    plt.axis('equal')
    plt.show()

    ax = world_map.visualize_passage(full_passage=False)

    for obs_vrtx in obs_vrtxs:
        print(obs_vrtx)
        for s in obs_vrtx:
            ax.scatter(s[0], s[1], c='r')
    plt.axis('equal')
    plt.show()

    load_mapCfg_to_mjcf(map_case, init_dlo_shape=planned_shape_seq[0], goal_dlo_shape=planned_shape_seq[-1])

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
        weight_curve = torch.diag(torch.tensor([1.0, 1.0] + [1.0, 1.0] * (13-2) + [1.0, 1.0], device=device)) * 150.0
        
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

                if sdf_val < 0.05:
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

                if sdf_val < 0.05:
                    obs_loss = obs_loss + relaxed_log_barrier(sdf_val-0.04, phi=phi)

        # eef_orientation
        left_theta = eef_states[0, 2]
        right_theta = eef_states[1, 2]
        left_theta_target = wrap_angle(angle_with_x_axis(target_dlo_kp_2d[1], target_dlo_kp_2d[0]))
        right_theta_target = wrap_angle(angle_with_x_axis(target_dlo_kp_2d[-1], target_dlo_kp_2d[-2]))
        # print(left_theta_target.data.item(), right_theta_target.data.item())

        # ctrl
        weight_u = torch.diag(torch.tensor([1.0, 1.0, 1.0] * 2, device=device)) * 1
        ctrl = u.flatten()
        
        ctrl_loss = 0.5 * (ctrl.T @ weight_u @ ctrl)

        orien_loss = (left_theta_target - left_theta) ** 2 + (right_theta_target - right_theta) ** 2 \
                     + 0# torch.sum((eef_states[0, 0:2] - target_dlo_kp_2d[0]) ** 2) + torch.sum((eef_states[1, 0:2] - target_dlo_kp_2d[-1]) ** 2)
        c = position_loss + curve_loss + ctrl_loss + 1e-2  * obs_loss + orien_loss * 0.5
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
        weight_curve = torch.diag(torch.tensor([1.0, 1.0] + [1.0, 1.0] * (13-2) + [1.0, 1.0], device=device)) * 150.0

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

                if sdf_val < 0.05:
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

                if sdf_val < 0.05:
                    obs_loss = obs_loss + relaxed_log_barrier(sdf_val-0.04, phi=phi) #1 / (sdf_val**4)
        
        orien_loss = (left_theta_target - left_theta) ** 2 + (right_theta_target - right_theta) ** 2 \
                      + 0# torch.sum((eef_states[0, 0:2] - target_dlo_kp_2d[0]) ** 2) + torch.sum((eef_states[1, 0:2] - target_dlo_kp_2d[-1]) ** 2)
        
        c = position_loss + curve_loss + orien_loss * 0.5 + 1e-2 * obs_loss
        return c


    # setup the simulation env
    render_mode = 'human' #'rgb_array' #  'depth_array' #
    camera = 'top_camera'
    task_id = '03'
    env = DualGripperCableEnv(task=task_id, feat_stride=3, render_mode=render_mode, camera_name=camera)
    obs, _ = env.reset()

    result_dir = '/home/yxtang/CodeBase/PythonCourse/PythonRobotics/PathPlanning/st_dlo_planning/results'
    video_res_path = os.path.join(result_dir, 'video', f'{map_case}.mp4')
    
    video_logger = misu.VideoLoggerPro(video_res_path, fps=int(3/env.dt))

    # env.step_relative_eef([0.02, 0.2, -0.0], [-0.02, 0.2, 0.0], num_steps=50, render_mode=render_mode)
    buffer = ReplayBuffer(obs_dim=env.num_feat*3 + env.num_grasp*3, 
                          act_dim=env.num_grasp*3)

    # ====================== global deformation model ===================
    model_dir = '/home/yxtang/CodeBase/PythonCourse/PythonRobotics/PathPlanning/st_dlo_planning/results/checkpoints/st_dlo_gdm_30/'
    model_ckpt_path = model_dir + 'latest.ckpt'

    gdm_model = GDM(GDM_CFG())

    model_params = torch.load(model_ckpt_path, pickle_module=dill)['state_dicts']['model']
    gdm_model.load_state_dict(model_params)
    gdm_model.to(device)

    lb = np.array([-0.04, -0.04, -0.4, -0.04, -0.04, -0.4]) / 3
    ub = np.array([ 0.04,  0.04,  0.4,  0.04,  0.04,  0.4]) / 3

    H = 1
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
                              discount_factor=0.95, pretrained_model=gdm_model,
                              traj_horizon=H, log_prefix='', mdl_ft=True) #False

    T = 7000
    error_list = []
    ultra_error_list = []
    action_list = []
    cons_vio_list = []
    time_list = []

    dlos = []
    target_dlos = []

    error_pre = 1e3
    obj_loss_pre = 1e7
    pre_action = 0.
    
    i = 0

    ref_idx = 0
    patience = 0
    error_patience = 0

    init_target_shape = planned_shape_seq[0].flatten()
    final_target_shape = planned_shape_seq[-1].flatten()
    ani_dlos = []

    while i < T:
        tc = time.time()

        dlo_kp = obs['dlo_keypoints']
        eef_states = obs['lowdim_eef_transforms']

        print(dlo_kp.shape, eef_states.shape)
        
        if ref_idx == 0:
            dlo_kp_ref = planned_shape_seq[0]
            delta_eef, obj_loss, hmin = agent.select_action(dlo_kp, eef_states, dlo_kp_ref, safe_mode=False)
        else:
            dlo_kp_ref = planned_shape_seq[ref_idx]
            # while True:
            #     dlo_kp_ref = planned_shape_seq[ref_idx]
            #     validate = check_shape_validality(dlo_kp_ref, world_map)
            #     if validate:
            #         break
            #     else:
            #         print('skipping invalidate reference shape!')
            #         ref_idx += 1
            delta_eef, obj_loss, hmin = agent.select_action(dlo_kp, eef_states, dlo_kp_ref, safe_mode=True)

        action = delta_eef

        action = np.clip(action, lb, ub)

        # action = 0.2 * pre_action + 0.8 * action
        if hmin > 0.02:
            num_step = 2
        else:
            num_step = 1
        next_obs, reward, done, truncated, info = env.step_relative_eef(action[0:3], action[3:], num_steps=num_step, render_mode=render_mode)
        buffer.store(obs, action, reward, next_obs, done) 
        img = env.render(mode=render_mode, camera_name=camera)

        error = np.linalg.norm(dlo_kp - dlo_kp_ref.flatten(), 2) / env.num_feat 
        # ================== model adaptation =================== #
        if error_pre - error < 0:
            b_s, b_a, b_r, b_ns, b_d = buffer.sample(128) 
            transition_dict = {'states': b_s, 'next_states': b_ns}
            update_step = 1
            # for _ in range(update_step):
            #     update_info = agent.model_adapt(transition_dict)
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
        ultra_error = np.linalg.norm(dlo_kp - final_target_shape, 2 ) / env.num_feat

        if i % 5 == 0:
            print(f'env_step: {i}, ref_idx {ref_idx}/{planned_shape_seq.shape[0]}, inter_err: {error:.4f}, ultra_err: {ultra_error:.4f}, hmin: {cons_vio:.4f}, patience: {patience}, err_patience: {error_patience}')

        if error >= error_pre and ref_idx > 0:
            error_patience += 1

        i += 1
        patience += 1

        if ref_idx > 0:
            ani_dlos.append(dlo_kp.reshape(-1, 3)[:, 0:2])
            if img is not None:
                video_logger.log_frame(img) 

        if (error < 4e-3 and ref_idx < planned_shape_seq.shape[0]-1) or (error_patience > 15) or (patience > 150 and ref_idx > 0):
            ref_idx += 1
            ref_idx = min(ref_idx, planned_shape_seq.shape[0] - 1)
            patience = 0
            error_patience = 0
        
        if ref_idx > 0:
            dlos.append(dlo_kp.reshape(-1, 3))
            target_dlos.append(dlo_kp_ref.reshape(-1, 3))
            error_list.append(error)
            ultra_error_list.append(ultra_error)
            action_list.append(action)
            cons_vio_list.append(cons_vio)
            
        if ultra_error <= 2e-3:
            dlos.append(dlo_kp.reshape(-1, 3))
            target_dlos.append(final_target_shape.reshape(-1, 3))
            break

        pre_action = action
        error_pre = error

    # ======================= plot ======================== #
    import matplotlib.pyplot as plt
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
    
    ax = world_map.visualize_passage(full_passage=False)

    for k in range(0, len(dlos), 150):
        dlo_shape = dlos[k]
        dlo_shape = dlo_shape.reshape(-1, 3)
        visualize_shape(dlo_shape, ax, clr='k', ld=1.5, s=10)
        # visualize_shape(target_dlos[k], ax, clr='r', ld=1.0)
    plt.axis('equal')
    plt.savefig(f"/home/yxtang/CodeBase/PythonCourse/PythonRobotics/PathPlanning/st_dlo_planning/results/tracking_res_{map_case}.png",
                dpi=1200)
    plt.show()

    fig_ani = plt.figure()
    ax_ani = fig_ani.add_subplot()
    world_map.visualize_passage(ax=ax_ani, full_passage=False)
    clrs = np.linspace(0.0, 1.0, len(ani_dlos))
    rever_clrs = np.flip(clrs)
    import matplotlib.animation as animation
    artists = []
    
    # init_target_shape = init_target_shape.reshape(-1, 3)
    # container_init = ax.plot(init_target_shape[:, 0], init_target_shape[:, 1], 
    #                          color='k', linewidth=1.5, marker='o', mec='k', mfc='k', ms=4)
    final_target_shape = final_target_shape.reshape(-1, 3)
    ax_ani.plot(final_target_shape[:, 0], final_target_shape[:, 1], 
                color='r', linewidth=1.5, marker='o', mec='r', mfc='r', ms=4)
    for i in range(0, len(ani_dlos), 3):
        container1 = ax_ani.plot(ani_dlos[i][:, 0], ani_dlos[i][:, 1], 
                             color=[clrs[i], rever_clrs[i], clrs[i]], linewidth=2)
        artists.append(container1)
        print(f'Frame: {i}')
    container1 = ax_ani.plot(ani_dlos[-1][:, 0], ani_dlos[-1][:, 1], 
                             color=[clrs[i], rever_clrs[i], clrs[i]], linewidth=2)
    artists.append(container1)
    containerf = ax_ani.plot(ani_dlos[-1][:, 0], ani_dlos[-1][:, 1], 
                             color=[clrs[-1], rever_clrs[-1], clrs[-1]], linewidth=1.5, marker='o', mec='r', mfc='r', ms=4)
    artists.append(containerf)
    
    plt.axis('equal')
    ani = animation.ArtistAnimation(fig=fig_ani, artists=artists, interval=20)
    ani.save(filename=f"{result_dir}/tracking_{map_case}.gif", writer="pillow")

    
    # save all the workspace results
    if save:
        if video_logger._frames[0] is not None:
            video_logger.create_video()

        import pickle
        ws_dir = '/home/yxtang/CodeBase/PythonCourse/PythonRobotics/PathPlanning/st_dlo_planning/results/exp_ws'
        file_to_save = os.path.join(ws_dir, f'{map_case}.pkl')
        f = open(file_to_save, "wb")
        res = {
            'error_list': error_list,
            'ultra_error_list': ultra_error_list,
            'action_list': action_list,
            'cons_vio_list': cons_vio_list,
            'dlos': dlos,
            'target_dlos': target_dlos,
            'reference_shapes': planned_shape_seq
        }
        pickle.dump(res, f)

        # close file
        f.close()