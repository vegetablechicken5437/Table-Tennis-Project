import numpy as np
from scipy.stats import gaussian_kde
from scipy.integrate import solve_ivp

def find_best_rps(rps_list):
    if len(rps_list) == 1:
        return rps_list[0]
    kde = gaussian_kde(rps_list)  # 進行 KDE（核密度估計）
    xs = np.linspace(min(rps_list), max(rps_list), 1000)    # 建立評估點
    kde_values = kde(xs)
    rps = xs[np.argmax(kde_values)]  # 找出 KDE peak 對應的角速度（即眾數）
    return rps

def calc_candidate_spin_rates(traj_3D, marks_3D, plane_normal, fps=225):
    """
    根據 traj_3D 和 marks_3D 計算四種候選旋轉速率 (RPS)，並根據正確方向分類
    - traj_3D: (N, 3) 的球心座標
    - marks_3D: (N, 3) 的標記座標
    - plane_normal: (3,) 的旋轉平面法向量
    - fps: 每秒幾張影格（預設 225）
    回傳：
    - rps_cw_list: 正確為順時針的 RPS
    - rps_cw_extra_list: 順時針 +360° 的 RPS
    - rps_ccw_list: 正確為逆時針的 RPS
    - rps_ccw_extra_list: 逆時針 +360° 的 RPS
    - valid_data: (i, v1, v2, theta_deg) 連續偵測到標記的 frame 資訊
    """

    delta_t = 1 / fps
    translated_marks = marks_3D - traj_3D
    projected_marks = translated_marks - np.outer(np.dot(translated_marks, plane_normal), plane_normal)

    rps_cw_list = []
    rps_cw_extra_list = []
    rps_ccw_list = []
    rps_ccw_extra_list = []

    valid_data = []

    for i in range(len(projected_marks) - 1):
        v1 = projected_marks[i]
        v2 = projected_marks[i + 1]

        if np.any(np.isnan(v1)) or np.any(np.isnan(v2)):
            continue

        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            continue

        cos_theta = np.clip(np.dot(v1, v2) / (norm_v1 * norm_v2), -1.0, 1.0)
        theta = np.arccos(cos_theta)
        theta_deg = theta * 360 / (2 * np.pi)

        cross = np.cross(v1, v2)
        direction = np.dot(cross, plane_normal)

        if direction < 0:
            theta_cw = theta_deg
            theta_ccw = 360 - theta_deg
        else:
            theta_ccw = theta_deg
            theta_cw = 360 - theta_deg

        rps_cw_list.append(round((theta_cw / 360) / delta_t, 4))
        rps_cw_extra_list.append(round(((theta_cw + 360) / 360) / delta_t, 4))
        rps_ccw_list.append(round((theta_ccw / 360) / delta_t, 4))
        rps_ccw_extra_list.append(round(((theta_ccw + 360) / 360) / delta_t, 4))

        valid_data.append((i, v1, v2, theta_deg))
    
    return (rps_cw_list, rps_cw_extra_list, rps_ccw_list, rps_ccw_extra_list), valid_data

def compute_traj_continuous(traj_3D_ori, dt, fps, aero_params, rps, spin_axis):
    g, m, rho, A, r, Cd, Cm = aero_params.values()

    v_init = (traj_3D_ori[1] - traj_3D_ori[0]) * fps
    x_init = traj_3D_ori[0]
    omega = rps * 2 * np.pi
    omega_vec = omega * spin_axis

    def dynamics(t, y):
        pos = y[:3]
        vel = y[3:]

        v_mag = np.linalg.norm(vel)
        if v_mag < 1e-6:
            return np.concatenate((vel, np.zeros(3)))

        Fd = -0.5 * Cd * rho * A * v_mag * vel
        Fm = Cm * rho * A * r * np.cross(omega_vec, vel)
        Fg = np.array([0, 0, -m * g])

        acc = (Fd + Fm + Fg) / m
        return np.concatenate((vel, acc))

    # 設定總模擬時間
    total_time = dt * (len(traj_3D_ori) - 1)
    t_span = (0, total_time)
    y0 = np.concatenate((x_init, v_init))

    t_eval = np.linspace(0, total_time, len(traj_3D_ori))  # 對齊原始軌跡幀數
    sol = solve_ivp(dynamics, t_span, y0, t_eval=t_eval, method='RK45')

    return sol.y[:3].T  # 回傳位置 (N, 3)

def compute_angular_velocity_rps(t, px, py, pz, aero_params):

    g, m, rho, A, r, Cd, Cm = aero_params.values()

    dx = px.deriv(1)(t)
    dy = py.deriv(1)(t)
    dz = pz.deriv(1)(t)
    vx = np.stack([dx, dy, dz], axis=1)

    ddx = px.deriv(2)(t)
    ddy = py.deriv(2)(t)
    ddz = pz.deriv(2)(t)
    ax = np.stack([ddx, ddy, ddz], axis=1)

    vnorm = np.linalg.norm(vx, axis=1)
    Fd = -0.5 * Cd * rho * A * vnorm[:, None] * vx
    Fnet = m * ax - m * g - Fd

    omega = np.cross(Fnet, vx) / (vnorm[:, None] ** 2)
    omega *= 3 / (4 * Cm * np.pi * r**3 * rho)
    omega[np.isnan(omega)] = 0

    rps = np.linalg.norm(omega, axis=1) / (2 * np.pi)

    # Normalize to get rotation axis (unit vector)
    axis_norm = np.linalg.norm(omega, axis=1, keepdims=True)
    axis_norm[axis_norm == 0] = 1  # avoid divide-by-zero
    rotation_axis = omega / axis_norm

    return rps, rotation_axis

def write_spin_info(output_path, seg_num, mark_detect_rate, valid_mark_frames, v_avg, results):
    with open(output_path, "w") as f:
        f.write(f"\n======= Trajectory Segment {seg_num+1} =======\n")
        f.write(f"Mark detection successful rate: {mark_detect_rate}\n")
        f.write(f"Valid mark frame count: {len(valid_mark_frames)}\n")
        f.write(f"Average Speed: {v_avg} m/s\n")
        f.write("===================================\n")

        for j in range(len(results)):
            candidate_round, spin_axis, best_rps, candidate_traj = results[j]
            f.write(f"Candidate_{candidate_round}: \n")
            f.write(f"Spin Axis: {list(map(lambda x: round(x, 4), spin_axis))}\n")
            f.write(f"Spin Rate: {round(best_rps, 4)} RPS\n")
            f.write("===================================\n")