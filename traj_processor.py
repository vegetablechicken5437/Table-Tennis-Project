import cv2
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN

# 簡易 Kalman filter：使用 constant velocity model 進行平滑
def simple_kalman_filter_3d(trajectory, FPS, R=25.0, Q=1.0):
    n = len(trajectory)
    dt = 1 / FPS  # 假設等間隔

    # 初始狀態 [x, y, z, vx, vy, vz]
    x = np.hstack((trajectory[0], [0, 0, 0]))
    P = np.eye(6) * 1000  # 初始協方差大

    # transition matrix F
    F = np.eye(6)
    for i in range(3):
        F[i, i+3] = dt

    # observation matrix H
    H = np.zeros((3, 6))
    H[0, 0] = 1
    H[1, 1] = 1
    H[2, 2] = 1

    # noise matrices
    R_matrix = np.eye(3) * R
    Q_matrix = np.eye(6) * Q

    smoothed = []
    for z in trajectory:
        # predict
        x = F @ x
        P = F @ P @ F.T + Q_matrix

        # update
        y = z - H @ x
        S = H @ P @ H.T + R_matrix
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ y
        P = (np.eye(6) - K @ H) @ P

        smoothed.append(x[:3])
    return np.array(smoothed)

def remove_outliers_with_pca(points, contamination=0.02):
    """
    對3D點雲資料執行主成分分析與離群值剔除，並儲存與回傳過濾後的結果。
    """
    # PCA（用於降維後再進行離群值偵測）
    pca = PCA(n_components=3)
    points_pca = pca.fit_transform(points)

    # 離群值偵測
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    outliers = iso_forest.fit_predict(points_pca)
    filtered_points = points[outliers == 1]

    # # 視覺化原始與過濾後的結果
    # fig = plt.figure(figsize=(12, 6))

    # # 原始資料
    # ax1 = fig.add_subplot(121, projection='3d')
    # ax1.scatter(points[:, 0], points[:, 1], points[:, 2], s=5, label='Original')
    # ax1.set_title("Original 3D Points")

    # # 過濾後資料
    # ax2 = fig.add_subplot(122, projection='3d')
    # ax2.scatter(filtered_points[:, 0], filtered_points[:, 1], filtered_points[:, 2], s=5, c='g', label='Filtered')
    # ax2.set_title("Filtered (Outliers Removed)")

    # plt.tight_layout()
    # plt.show()

    return filtered_points

def remove_outliers_stronger(points, contamination=0.02, std_threshold=2.5):
    """
    強化版離群值剔除：先用 PCA + IsolationForest，再用 Z-score 移除極端值。

    Parameters:
        input_path (str): 輸入txt路徑
        output_path (str): 儲存路徑（預設為 *_filtered_strict.txt）
        contamination (float): IsolationForest 的離群比例
        std_threshold (float): Z-score 標準差門檻（預設為2.5）
        save_result (bool): 是否儲存檔案

    Returns:
        numpy.ndarray: 最終過濾後點雲
    """

    # 第一步：PCA + IsolationForest
    pca = PCA(n_components=1)
    pca_transformed = pca.fit_transform(points)

    iso = IsolationForest(contamination=contamination, random_state=42)
    mask_iforest = iso.fit_predict(pca_transformed) == 1
    filtered = points[mask_iforest]

    # 第二步：Z-score 清除極端值
    mean = np.mean(filtered, axis=0)
    std = np.std(filtered, axis=0)
    z_scores = np.abs((filtered - mean) / std)
    mask_zscore = np.all(z_scores < std_threshold, axis=1)
    final_filtered = filtered[mask_zscore]

    return final_filtered

def remove_outliers_by_dbscan(traj_3D, eps=10, min_samples=5):
    traj_3D = np.array(traj_3D)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(traj_3D)
    labels = clustering.labels_
    main_label = np.argmax(np.bincount(labels[labels >= 0]))  # 取得最大群
    mask = labels == main_label
    return traj_3D[mask], mask

def remove_outliers_by_speed(traj_3D, max_speed_threshold=30):
    traj_3D = np.array(traj_3D)
    diffs = np.linalg.norm(np.diff(traj_3D, axis=0), axis=1)
    mask = np.ones(len(traj_3D), dtype=bool)
    mask[1:] &= diffs < max_speed_threshold
    mask[:-1] &= diffs < max_speed_threshold
    return traj_3D[mask], mask

def detect_table_tennis_collisions(traj, table_corners, z_tolerance=0.05):
    """
    偵測桌球最多三次碰撞點：第一次（落桌）、第二次（擊球）、第三次（回擊）
    條件：
        - 撞擊點需在球桌範圍內
        - Z 軸需接近桌面高度
        - 根據 Y 軸進行分區：前後場與拍擊位置

    :param traj: (N, 3) 的軌跡資料 (X, Y, Z)
    :param table_corners: (4, 3) 的球桌四角座標
    :param z_tolerance: 與桌面高度的容許誤差
    :return: [(idx1, point1), (idx2, point2), (idx3, point3)] (最多三項)
    """
    traj = np.array(traj)
    table_corners = np.array(table_corners)

    # 桌面範圍與高度估計
    x_min, x_max = table_corners[:, 0].min(), table_corners[:, 0].max()
    y_min, y_max = table_corners[:, 1].min(), table_corners[:, 1].max()
    z_table = table_corners[:, 2].mean()
    net_y = (y_min + y_max) / 2

    def is_near_table(pos):
        x, y, z = pos
        return (x_min <= x <= x_max) and (y_min <= y <= y_max) and (abs(z - z_table) < z_tolerance)

    # 第一次碰撞：網子對面半場 z 最小
    first_candidates = [(i, p[2]) for i, p in enumerate(traj) if p[1] > net_y and is_near_table(p)]
    first_hit = min(first_candidates, key=lambda x: x[1])[0] if first_candidates else None

    # 第二次碰撞：Y 最大點（擊球），要在 first_hit 之後
    second_hit = None
    if first_hit is not None:
        y_vals_after_first = traj[first_hit + 1:, 1]
        if len(y_vals_after_first) > 0:
            second_hit_rel = np.argmax(y_vals_after_first)
            second_hit = first_hit + 1 + second_hit_rel

    # 第三次碰撞：網子這側 z 最小，要在 second_hit 之後
    third_hit = None
    if second_hit is not None:
        third_candidates = [(i, p[2]) for i, p in enumerate(traj[second_hit + 1:], start=second_hit + 1)
                            if p[1] < net_y and is_near_table(p)]
        if third_candidates:
            third_hit = min(third_candidates, key=lambda x: x[1])[0]

    # 整理回傳 (index, point)
    collision_indices = [first_hit, second_hit, third_hit]
    results = [(idx, traj[idx]) for idx in collision_indices if idx is not None]
    return results

def split_trajectory_by_collisions(traj, collisions):
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

    if start_idx < len(traj):
        segments.append(traj[start_idx:])  # 剩下的當作最後一段

    return segments

if __name__ == "__main__":
    pass