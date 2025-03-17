import numpy as np
from visualize_functions import draw_trajectories

def fit_spin_axis(ball_frame_nums, logo_frame_nums, traj_3D, logos_3D):
    logo_indices = np.array([np.where(ball_frame_nums == f)[0][0] for f in logo_frame_nums])    # 找到 logo_frame_nums 在 ball_frame_nums 中的索引
    ball_centers = traj_3D[logo_indices]    # 提取對應的球心座標
    translated_logos = logos_3D - ball_centers  # 計算 logo 相對於球心的位置 (球心到 logo 向量)
    distances = np.linalg.norm(translated_logos, axis=1)    # 計算加權SVD，權重與距離原點成反比
    weights = 1 / (distances + 1e-6)  # 避免除以0

    # 計算加權中心
    weighted_center = np.average(translated_logos, axis=0, weights=weights)
    centered_logos = translated_logos - weighted_center

    # SVD 擬合平面
    U, S, Vt = np.linalg.svd(centered_logos * weights[:, np.newaxis])
    plane_normal = Vt[-1]  # 確保選取正交於平面的法向量
    plane_normal /= np.linalg.norm(plane_normal)  # 單位化

    return translated_logos, plane_normal

def calc_candidate_spin_rates(ball_frame_nums, logo_frame_nums, traj_3D, logos_3D, plane_normal):
    # 設定影片幀率（FPS）
    fps = 214
    delta_t = 1 / fps  # 每幀的時間間隔 (秒)

    # 找到 logo_frame_nums 在 ball_frame_nums 中的索引
    logo_indices = np.array([np.where(ball_frame_nums == f)[0][0] for f in logo_frame_nums])    
    ball_centers = traj_3D[logo_indices]    # 提取對應的球心座標
    translated_logos = logos_3D - ball_centers  # 計算 logo 相對於球心的位置 (球心到 logo 向量)

    # 計算每個 logo 位置相對於旋轉軸（平面法向量）的投影向量
    projected_logos = translated_logos - np.outer(np.dot(translated_logos, plane_normal), plane_normal)

    # 計算相鄰兩個 logo 位置的變化向量
    logo_vectors = np.diff(projected_logos, axis=0)

    # 計算每個相鄰 logo 偵測點之間的時間間隔 Δt
    delta_t_list = np.diff(logo_frame_nums) / fps  # 每組相鄰點的時間間隔 (秒)

    # 計算旋轉角速度 (RPS)
    rps_cw_list = []
    rps_cw_extra_list = []
    rps_ccw_list = []
    rps_ccw_extra_list = []

    for i in range(len(logo_vectors) - 1):
        v1 = projected_logos[i]  # 從旋轉軸到 logo 位置的距離向量
        v2 = projected_logos[i + 1]

        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 > 0 and norm_v2 > 0:  # 避免除以零
            cos_theta = np.clip(np.dot(v1, v2) / (norm_v1 * norm_v2), -1.0, 1.0)
            theta = np.arccos(cos_theta)  # 角度 (弧度)
            
            # 計算四種情況
            theta_cw = theta  # 順時針
            theta_cw_extra = theta + 2 * np.pi  # 順時針多一圈
            theta_ccw = 2 * np.pi - theta  # 逆時針
            theta_ccw_extra = 2 * np.pi + (2 * np.pi - theta)  # 逆時針多一圈

            # 對應 Δt 計算 RPS
            if delta_t_list[i] == delta_t:
                rps_cw_list.append(theta_cw / (2 * np.pi) / delta_t_list[i])
                rps_cw_extra_list.append(theta_cw_extra / (2 * np.pi) / delta_t_list[i])
                rps_ccw_list.append(theta_ccw / (2 * np.pi) / delta_t_list[i])
                rps_ccw_extra_list.append(theta_ccw_extra / (2 * np.pi) / delta_t_list[i])

    # 計算平均 RPS
    rps_cw = np.mean(rps_cw_list)
    rps_cw_extra = np.mean(rps_cw_extra_list)
    rps_ccw = np.mean(rps_ccw_list)
    rps_ccw_extra = np.mean(rps_ccw_extra_list)

    return rps_cw, rps_cw_extra, rps_ccw, rps_ccw_extra

def compute_trajectory_aero(v_init, x_init, rps, dt_list, num_steps, plane_normal, aero_params):
    """
    使用空氣動力學公式計算桌球的軌跡，考慮空氣阻力與馬格努斯力。
    """
    g, m, rho, A, r, Cd, Cm = aero_params.values()

    pos = np.zeros((num_steps, 3))
    vel = np.nan_to_num(v_init, nan=0.01, posinf=0.01, neginf=-0.01)
    pos[0] = x_init

    for i in range(1, num_steps - 1):
        dt = dt_list[i - 1] if i < len(dt_list) else dt_list[-1]

        v_mag = np.linalg.norm(vel)
        if v_mag < 1e-6:
            continue

        # 計算空氣阻力 (Drag Force)
        Fd = -0.5 * Cd * rho * A * v_mag * vel

        # 計算馬格努斯力 (Magnus Force)
        omega = rps * 2 * np.pi     # 轉速（RPS）轉換為角速度 ω (rad/s)
        omega_vec = omega * plane_normal
        Fm = Cm * rho * A * r * np.cross(omega_vec, vel)
        
        if np.isnan(Fm).any() or np.isinf(Fm).any():
            Fm = np.zeros(3)

        # 總合力
        F_total = Fd + Fm + np.array([0, 0, -m * g])

        # 計算加速度並更新速度與位置
        acc = np.nan_to_num(F_total / m, nan=0, posinf=0, neginf=0)
        vel += acc * dt
        pos[i] = pos[i - 1] + vel * dt

    return pos[:-1]

def find_best_matching_rps(traj_3D, trajectories, rps_values):
    """
    比較不同計算軌跡與 Ground Truth (traj_3D) 的差異，選擇最匹配的旋轉速度 (RPS)。
    
    :param traj_3D: Ground Truth 軌跡 (numpy array)
    :param trajectories: 字典，包含不同旋轉情況的計算軌跡
    :param rps_values: 對應的 RPS 數值 (dict)
    :return: 最小誤差的 RPS 數值
    """
    traj_3D = traj_3D[:-1]
    min_error = float('inf')
    best_rps = None

    for label, trajectory in trajectories.items():
        # 計算與 Ground Truth 的平均誤差 (歐幾里得距離)
        error = np.mean(np.linalg.norm(traj_3D[:len(trajectory)] - trajectory, axis=1))

        # 找到最小誤差的 RPS
        if error < min_error:
            min_error = error
            best_rps = rps_values[label]

    return best_rps




if __name__ == "__main__":

    sample_num = 2
    sample_folder_name = f'sample-{sample_num}x'
    output_folder = f'OUTPUT/{sample_folder_name}'

    ball_frame_nums = np.loadtxt(f'{output_folder}/ball_frame_nums.txt', dtype=int)
    logo_frame_nums = np.loadtxt(f'{output_folder}/logo_frame_nums.txt', dtype=int)
    traj_3D = np.loadtxt(f'{output_folder}/traj_3D.txt')  # 完整球軌跡
    logos_3D = np.loadtxt(f'{output_folder}/logos_3D.txt')  # 偵測到logo的3D點

    plane_normal = fit_spin_axis(ball_frame_nums, logo_frame_nums, traj_3D, logos_3D)
    rps_cw, rps_cw_extra, rps_ccw, rps_ccw_extra = calc_candidate_spin_rates(ball_frame_nums, logo_frame_nums, traj_3D, logos_3D, plane_normal)

    # 空氣動力學參數: [重力加速度 (m/s^2), 桌球質量 (kg), 空氣密度 (kg/m^3), 球的迎風面積 (m^2), 球半徑 (m), 阻力係數, 馬格努斯力係數]
    aero_params = {'g':9.8, 'm':0.0027, 'rho':1.2, 'A':0.001256, 'r':0.02, 'Cd':0.5, 'Cm':1.23}

    # 計算每一幀的速度 (Ground Truth)
    fps = 214
    dt_list = np.diff(ball_frame_nums) / fps  # 每一幀的時間間隔 (秒)
    velocity_gt = np.diff(traj_3D, axis=0) * fps  # 速度計算
    acceleration_gt = np.diff(velocity_gt, axis=0) * fps  # 加速度計算

    # 設定模擬步數
    num_steps = len(traj_3D)

    # 計算四種旋轉條件的軌跡
    trajectory_cw = compute_trajectory_aero(velocity_gt[0], traj_3D[0], rps_cw, dt_list, num_steps, plane_normal, aero_params)
    trajectory_cw_extra = compute_trajectory_aero(velocity_gt[0], traj_3D[0], rps_cw_extra, dt_list, num_steps, plane_normal, aero_params)
    trajectory_ccw = compute_trajectory_aero(velocity_gt[0], traj_3D[0], rps_ccw, dt_list, num_steps, plane_normal, aero_params)
    trajectory_ccw_extra = compute_trajectory_aero(velocity_gt[0], traj_3D[0], rps_ccw_extra, dt_list, num_steps, plane_normal, aero_params)

    draw_trajectories(traj_3D, trajectory_cw, trajectory_cw_extra, trajectory_ccw, trajectory_ccw_extra, "candidate_trajectories.html")

    print("Corrected Rotation Axis (Plane Normal):", plane_normal)
    print(rps_cw, rps_cw_extra, rps_ccw, rps_ccw_extra)

    