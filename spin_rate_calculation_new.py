import numpy as np
from scipy.stats import gaussian_kde

def find_best_rps(rps_list):
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
    """

    delta_t = 1 / fps
    translated_logos = marks_3D - traj_3D
    projected_logos = translated_logos - np.outer(np.dot(translated_logos, plane_normal), plane_normal)

    rps_cw_list = []
    rps_cw_extra_list = []
    rps_ccw_list = []
    rps_ccw_extra_list = []

    for i in range(len(projected_logos) - 1):
        v1 = projected_logos[i]
        v2 = projected_logos[i + 1]

        if np.any(np.isnan(v1)) or np.any(np.isnan(v2)):
            continue

        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            continue

        cos_theta = np.clip(np.dot(v1, v2) / (norm_v1 * norm_v2), -1.0, 1.0)
        theta = np.arccos(cos_theta)
        theta = theta * 360 / (2 * np.pi)

        # 判斷方向
        cross = np.cross(v1, v2)
        direction = np.dot(cross, plane_normal)

        # 根據方向切換夾角定義
        if direction < 0:  # 順時針
            theta_cw = theta
            theta_ccw = 360 - theta
        else:  # 逆時針或無方向（也當逆）
            theta_ccw = theta
            theta_cw = 360 - theta

        # print('theta: ', theta)
        # print('theta_cw: ', theta_cw)
        # print('theta_ccw: ', theta_ccw)

        # 計算 RPS
        rps_cw_list.append(round((theta_cw / 360) / delta_t, 4))        # 角度除360表示幾圈
        rps_cw_extra_list.append(round(((theta_cw + 360) / 360) / delta_t, 4))
        rps_ccw_list.append(round((theta_ccw / 360) / delta_t, 4))
        rps_ccw_extra_list.append(round(((theta_ccw + 360) / 360) / delta_t, 4))

    return rps_cw_list, rps_cw_extra_list, rps_ccw_list, rps_ccw_extra_list


def trimmed_mean_rps(candidates, trim_frac=0.1):
    arr = np.sort(np.array(candidates))
    n = len(arr)
    k = int(n * trim_frac)
    if n > 2*k:
        arr = arr[k:-k]
    return np.nanmean(arr)

def compute_trajectory_aero(v_init, x_init, rps, dt, num_steps, plane_normal, aero_params):
    """
    使用空氣動力學公式計算桌球的軌跡，考慮空氣阻力與馬格努斯力。
    """
    g, m, rho, A, r, Cd, Cm = aero_params.values()

    pos = np.zeros((num_steps, 3))
    vel = np.nan_to_num(v_init, nan=0.01, posinf=0.01, neginf=-0.01)
    pos[0] = x_init

    for i in range(1, num_steps - 1):

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
