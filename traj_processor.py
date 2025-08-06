import cv2
import numpy as np
from scipy.interpolate import interp1d
from pykalman import KalmanFilter
from numpy.polynomial.polynomial import Polynomial
from pykalman import KalmanFilter
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

def detect_table_tennis_collisions(traj, table_corners, z_tolerance=0.05):
    """
    更嚴謹的三次碰撞點偵測：根據 Z 軌跡的先下降再上升的轉折點為主
    條件：
        - 撞擊點需在球桌範圍內
        - Z 軸需接近桌面高度
        - 依照順序，符合反彈形狀的點才能視為合法碰撞點
        - Y 軸範圍劃分前後場與拍擊段落
    :param traj: (N, 3) 的軌跡資料 (X, Y, Z)
    :param table_corners: (4, 3) 的球桌四角座標
    :param z_tolerance: 與桌面高度的容許誤差
    :return: [(idx1, point1), (idx2, point2), (idx3, point3)] (最多三項)
    """
    traj = np.array(traj)
    z = traj[:, 2]
    y = traj[:, 1]

    table_corners = np.array(table_corners)
    x_min, x_max = table_corners[:, 0].min(), table_corners[:, 0].max()
    y_min, y_max = table_corners[:, 1].min(), table_corners[:, 1].max()
    z_table = table_corners[:, 2].mean()
    net_y = (y_min + y_max) / 2

    def is_near_table(pos):
        x, y_, z_ = pos
        return (x_min <= x <= x_max) and (y_min <= y_ <= y_max) and (abs(z_ - z_table) < z_tolerance)

    collisions = []
    stage = 0  # 0: first bounce, 1: strike, 2: return bounce

    for i in range(1, len(traj) - 1):
        prev_z, this_z, next_z = z[i - 1], z[i], z[i + 1]
        this_y = y[i]
        if this_z < prev_z and this_z < next_z and is_near_table(traj[i]):
            if stage == 0 and this_y > net_y:
                collisions.append((i, traj[i]))
                stage = 1
            elif stage == 2 and this_y < net_y:
                collisions.append((i, traj[i]))
                stage = 3  # finish
        elif stage == 1:
            # find peak Y (hit)
            if this_y > y[i - 1] and this_y > y[i + 1]:
                collisions.append((i, traj[i]))
                stage = 2

    return collisions

def split_traj_by_collisions(traj, collisions):
    """
    根據碰撞點 index 分割軌跡段落，最多切成四段
    :param traj: 原始軌跡資料 (N, 3)
    :param collisions: [(index, point), ...] 最多三個碰撞點
    :return: List of trajectory segments (List of np.ndarray)
    """
    traj = np.array(traj)
    indices = [idx for idx, _ in collisions]
    segments = []

    start_idx = 0
    for idx in indices:
        if idx is not None and start_idx <= idx:
            segments.append(traj[start_idx:idx + 1])  # 包含碰撞點
            start_idx = idx + 1

    if start_idx < len(traj) - 10:  # 如果start_idx接近尾端 就不要插入後續軌跡
        segments.append(traj[start_idx:])  # 剩下的當作最後一段

    return segments

def remove_outliers_by_knn_distance(traj, k=5, sigma_thres=3.0):
    """
    根據與最近鄰點的平均距離找離群點（空間孤立性），並將離群點設為 NaN。

    Args:
        traj (ndarray): shape (N, 3)，含 NaN 的 3D 軌跡資料。
        k (int): 最近鄰數量（包含自己，預設5）。
        sigma_thres (float): 超過 mean + sigma_thres * std 的點視為離群。

    Returns:
        cleaned_traj (ndarray): 相同 shape，離群點設為 [nan, nan, nan]。
        outlier_indices (list): 離群點在原始資料中的索引。
    """
    traj = np.array(traj)
    valid_mask = ~np.isnan(traj).any(axis=1)
    valid_points = traj[valid_mask]
    valid_indices = np.where(valid_mask)[0]

    # 計算每個點與其 k-1 個鄰居的距離（排除自己）
    nbrs = NearestNeighbors(n_neighbors=k).fit(valid_points)
    distances, _ = nbrs.kneighbors(valid_points)
    mean_neighbor_dist = distances[:, 1:].mean(axis=1)

    # 找出離群門檻
    dist_mean = mean_neighbor_dist.mean()
    dist_std = mean_neighbor_dist.std()
    threshold = dist_mean + sigma_thres * dist_std

    # 找出離群點索引
    outlier_local_idx = np.where(mean_neighbor_dist > threshold)[0]
    outlier_indices = valid_indices[outlier_local_idx]

    # 清除離群點
    cleaned_traj = traj.copy()
    cleaned_traj[outlier_indices] = [np.nan, np.nan, np.nan]

    return cleaned_traj, outlier_indices.tolist()

def kalman_smooth_with_interp(traj_3D, smooth_strength=1.0, extend_points=5, dt=1.0):
    """
    對3D軌跡進行線性內插與卡爾曼平滑，並在首尾延伸虛擬點減少邊界效應。

    Parameters:
    - traj_3D: shape (N, 3) 的陣列，可能含有 NaN。
    - smooth_strength: 平滑強度，越大代表越平滑。
    - extend_points: 首尾要延伸的虛擬點數量。
    - dt: 每個時間步長，預設為 1.0。

    Returns:
    - smoothed_traj: 經平滑後的3D軌跡，長度與原始輸入相同。
    """
    traj_3D = np.array(traj_3D, dtype=np.float64)
    mask = ~np.isnan(traj_3D).any(axis=1)

    full_indices = np.arange(len(traj_3D))
    interp_traj = np.empty_like(traj_3D)

    for i in range(3):
        valid_idx = full_indices[mask]
        valid_vals = traj_3D[mask, i]
        interp_traj[:, i] = np.interp(full_indices, valid_idx, valid_vals)

    # 延伸首尾（可留原邏輯）
    head_dir = np.mean(interp_traj[1:6] - interp_traj[0:5], axis=0)
    tail_dir = np.mean(interp_traj[-5:] - interp_traj[-6:-1], axis=0)
    noise = np.random.normal(scale=0.001, size=(extend_points, 3))
    head_extend = [interp_traj[0] - head_dir * (i+1) + noise[i] for i in range(extend_points)][::-1]
    tail_extend = [interp_traj[-1] + tail_dir * (i+1) + noise[i] for i in range(extend_points)]
    extended_traj = np.vstack([head_extend, interp_traj, tail_extend])

    # === transition matrix 加入速度模型 ===
    F = np.eye(6)
    F[0, 3] = F[1, 4] = F[2, 5] = dt

    # === observation matrix（只觀察位置）===
    H = np.hstack([np.eye(3), np.zeros((3, 3))])

    # === 設定 initial_state，速度用首兩點估算 ===
    init_vel = (extended_traj[1] - extended_traj[0]) / dt
    initial_state_mean = np.hstack([extended_traj[0], init_vel])

    # === 放寬 transition_cov，提升靈活度 ===
    transition_cov = np.diag([1e-3 * smooth_strength]*3 + [1e-4 * smooth_strength]*3)
    observation_cov = np.eye(3) * 1e-2 / smooth_strength

    kf = KalmanFilter(
        transition_matrices=F,
        observation_matrices=H,
        transition_covariance=transition_cov,
        observation_covariance=observation_cov,
        initial_state_mean=initial_state_mean,
        initial_state_covariance=np.eye(6) * 1e-1
    )

    smoothed_state_means, _ = kf.smooth(extended_traj)
    smoothed_traj = smoothed_state_means[:, :3]
    smoothed_traj = smoothed_traj[extend_points:-extend_points]

    smoothed_speed = smoothed_state_means[:, 3:]  # vx, vy, vz
    smoothed_velocity = np.linalg.norm(smoothed_speed[extend_points:-extend_points])

    return smoothed_traj, smoothed_velocity

def shift_marks_by_traj(original_traj, smoothed_traj, marks_3D):
    delta = smoothed_traj - original_traj
    shifted_marks = marks_3D.copy()
    mask = ~np.isnan(marks_3D).any(axis=1)
    shifted_marks[mask] += delta[mask]
    return shifted_marks

def extract_valid_traj(traj_3D):
    """
    從頭尾刪除全是nan的列，只保留有效的[x,y,z]軌跡部分
    :param traj_np: numpy array of shape (N, 3)
    :return: numpy array without leading/trailing nan rows
    """
    if traj_3D.size == 0:
        return traj_3D

    # 判斷每一列是不是全部都是nan
    valid = ~np.isnan(traj_3D).all(axis=1)

    # 找到第一個True和最後一個True
    valid_indices = np.where(valid)[0]
    if valid_indices.size == 0:
        return np.empty((0, 3))  # 全部都是nan的情況

    start_idx = valid_indices[0]
    end_idx = valid_indices[-1]

    return traj_3D[start_idx:end_idx+1], start_idx, end_idx


if __name__ == "__main__":
    pass
