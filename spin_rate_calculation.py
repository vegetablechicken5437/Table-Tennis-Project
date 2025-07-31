import numpy as np
from scipy.stats import gaussian_kde
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

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


def trimmed_mean_rps(candidates, trim_frac=0.1):
    arr = np.sort(np.array(candidates))
    n = len(arr)
    k = int(n * trim_frac)
    if n > 2*k:
        arr = arr[k:-k]
    return np.nanmean(arr)

def compute_trajectory_aero(traj_3D_ori, dt, fps, aero_params, rps, spin_axis):
    """
    使用空氣動力學公式計算桌球的軌跡，考慮空氣阻力與馬格努斯力。
    """
    # 計算每一幀的速度 (Ground Truth)
    velocity_gt = np.diff(traj_3D_ori, axis=0) * fps    # 速度計算
    acceleration_gt = np.diff(velocity_gt, axis=0) * fps    # 加速度計算
    num_steps = len(traj_3D_ori)    # 設定模擬步數

    v_init, x_init = velocity_gt[0], traj_3D_ori[0]

    g, m, rho, A, r, Cd, Cm = aero_params.values()

    pos = np.zeros((num_steps, 3))
    vel = np.nan_to_num(v_init, nan=0.01, posinf=0.01, neginf=-0.01)
    pos[0] = x_init

    for i in range(1, num_steps - 1):

        v_mag = np.linalg.norm(vel)
        if v_mag < 1e-6:
            print(v_mag)
            continue

        # 計算空氣阻力 (Drag Force)
        Fd = -0.5 * Cd * rho * A * v_mag * vel

        # 計算馬格努斯力 (Magnus Force)
        omega = rps * 2 * np.pi     # 轉速（RPS）轉換為角速度 ω (rad/s)
        omega_vec = omega * spin_axis
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

def compute_trajectory_continuous(traj_3D_ori, dt, fps, aero_params, rps, spin_axis):
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

def optimize_spin_parameters(traj_3D_ori, dt, fps, aero_params,
                             rps_init=None, spin_axis_init=None, rps_delta=20):
    """
    根據初始 rps 和可接受變動範圍，優化旋轉軸與旋轉速度，使模擬軌跡最貼近真實軌跡。

    Parameters:
        traj_3D_ori: (N, 3) array 真實軌跡
        rps_init: float 初始旋轉速度（rps）
        dt: float 每幀間隔時間（1/fps）
        fps: int 每秒幾幀
        aero_params: dict 物理參數（g, m, rho, A, r, Cd, Cm）
        spin_axis_init: array-like 初始旋轉軸方向（可選），預設為 z 軸
        rps_delta: float rps 的調整範圍（會優化區間為 rps_init ± delta）

    Returns:
        best_spin_axis: ndarray 單位旋轉軸
        best_rps: float 最佳 rps
        min_error: float 最小平均距離誤差
    """

    if spin_axis_init is None:
        spin_axis_init = np.array([0, 0, 1])

    def loss_fn(params):
        theta, phi, rps = params
        spin_axis = np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])
        # sim_traj = compute_trajectory_aero(traj_3D_ori, dt, fps, aero_params, rps, spin_axis)
        sim_traj = compute_trajectory_continuous(traj_3D_ori, dt, fps, aero_params, rps, spin_axis)
        if np.any(np.isnan(sim_traj)):
            return np.inf
        error = np.mean(np.linalg.norm(sim_traj - traj_3D_ori[:len(sim_traj)], axis=1))
        return error

    # 將 spin_axis_init 轉為 theta/phi 角度
    x, y, z = spin_axis_init
    theta0 = np.arccos(z / np.linalg.norm(spin_axis_init))
    phi0 = np.arctan2(y, x)
    init = [theta0, phi0, rps_init]

    bounds = [
        (0, np.pi),                # θ: 0 ~ π
        (0, 2 * np.pi),            # φ: 0 ~ 2π
        (rps_init - rps_delta, rps_init + rps_delta)  # rps 微調範圍
    ]

    res = minimize(loss_fn, init, bounds=bounds, method='L-BFGS-B')

    theta_opt, phi_opt, rps_opt = res.x
    spin_axis_opt = np.array([
        np.sin(theta_opt) * np.cos(phi_opt),
        np.sin(theta_opt) * np.sin(phi_opt),
        np.cos(theta_opt)
    ])
    return spin_axis_opt, rps_opt, res.fun


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
