import numpy as np

def calc_candidate_spin_rates(FPS, traj_3D, marks_3D, spin_axis_vec):

    # print(len(traj_3D))
    # print(len(marks_3D[~np.isnan(marks_3D).any(axis=1)] ))

    # 單位化旋轉軸
    spin_axis = np.array(spin_axis_vec)
    spin_axis /= np.linalg.norm(spin_axis)

    # 收集有標記的 frame index
    # valid_idx = [
    #     i for i, m in enumerate(marks_3D)
    #     if m is not None and not np.allclose(m, 0)
    # ]
    valid_idx = [
        i for i, m in enumerate(marks_3D)
        if not np.isnan(m[0]) and not np.allclose(m, 0)
    ]

    # print(valid_idx)

    rps_cw, rps_cw_extra = [], []
    rps_ccw, rps_ccw_extra = [], []

    # 投影到旋轉平面的函式
    def project(v, n):
        return v - np.dot(v, n) * n

    for k in range(len(valid_idx) - 1):
        i, j = valid_idx[k], valid_idx[k+1]
        mark1, mark2 = marks_3D[i], marks_3D[j]

        # 取相對球心向量
        v1 = project(np.array(mark1) - np.array(traj_3D[i]), spin_axis)
        v2 = project(np.array(mark2) - np.array(traj_3D[j]), spin_axis)

        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            continue

        u1, u2 = v1/n1, v2/n2
        # 夾角
        dot = np.clip(np.dot(u1, u2), -1.0, 1.0)
        angle = np.arccos(dot)
        # 方向判斷
        dir_sign = np.dot(np.cross(u1, u2), spin_axis)
        is_cw = dir_sign < 0

        # 時間差 (秒)
        delta_t = (j - i) / FPS

        # 基本與多轉一圈 RPS
        base_rps  = angle / (2*np.pi * delta_t)
        extra_rps = (angle + 2*np.pi) / (2*np.pi * delta_t)

        if is_cw:
            rps_cw.append(base_rps)
            rps_cw_extra.append(extra_rps)
        else:
            rps_ccw.append(base_rps)
            rps_ccw_extra.append(extra_rps)

    return rps_cw, rps_cw_extra, rps_ccw, rps_ccw_extra

def trimmed_mean_rps(candidates, trim_frac=0.1):
    arr = np.sort(np.array(candidates))
    n = len(arr)
    k = int(n * trim_frac)
    if n > 2*k:
        arr = arr[k:-k]
    return round(np.nanmean(arr), 4)

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