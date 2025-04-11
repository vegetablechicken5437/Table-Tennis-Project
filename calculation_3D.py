import cv2
import numpy as np
import os
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

def read_calibration_file(path):
    data = {}
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith('#'):
                key = line[1:]
                values = []
            else:
                li = [float(x) for x in line.split()]
                if li:
                    values.append(li)
                else:
                    data[key] = np.array(values)
    return data

def get_valid_LR_ball_centers(LR_map, all_2D_centers):
    left_pts = []
    right_pts = []

    for base in sorted(LR_map.keys()):
        pair = LR_map[base]
        if pair["L"] is not None and pair["R"] is not None:
            L_file = pair["L"]
            R_file = pair["R"]

            if 0 in all_2D_centers[L_file] and 0 in all_2D_centers[R_file]:
                left_pts.append(all_2D_centers[L_file][0])
                right_pts.append(all_2D_centers[R_file][0])

    return left_pts, right_pts

def get_LR_centers_with_marks(LR_map, all_2D_centers):
    """
    根據 all_2D_centers 建立包含球與標記資訊的結構，
    並分別輸出對應的座標 list，長度一致。
    """
    # 檢查每個 L-R 配對是否都有球，並加入 mark 資訊
    valid_LR_centers = []

    for base in sorted(LR_map.keys()):
        pair = LR_map[base]
        L_file, R_file = pair["L"], pair["R"]

        if L_file and R_file:
            if 0 in all_2D_centers[L_file] and 0 in all_2D_centers[R_file]:
                entry = {"L": {}, "R": {}}

                # Ball center
                entry["L"]["ball"] = all_2D_centers[L_file][0]
                entry["R"]["ball"] = all_2D_centers[R_file][0]

                # Optional mark_o (key = 1)
                entry["L"]["mark_o"] = all_2D_centers[L_file].get(1, None)
                entry["R"]["mark_o"] = all_2D_centers[R_file].get(1, None)

                # Optional mark_x (key = 2)
                entry["L"]["mark_x"] = all_2D_centers[L_file].get(2, None)
                entry["R"]["mark_x"] = all_2D_centers[R_file].get(2, None)

                valid_LR_centers.append(entry)

    # 分別組出六個 list
    left_balls     = [item["L"]["ball"] for item in valid_LR_centers]
    right_balls    = [item["R"]["ball"] for item in valid_LR_centers]
    left_mark_o    = [item["L"].get("mark_o", None) for item in valid_LR_centers]
    right_mark_o   = [item["R"].get("mark_o", None) for item in valid_LR_centers]
    left_mark_x    = [item["L"].get("mark_x", None) for item in valid_LR_centers]
    right_mark_x   = [item["R"].get("mark_x", None) for item in valid_LR_centers]

    return left_balls, right_balls, left_mark_o, right_mark_o, left_mark_x, right_mark_x

def myDLT(camParams, left_pts, right_pts):
    left_camera_P = np.dot(camParams['LeftCamK'], camParams['LeftCamRT'])
    right_camera_P = np.dot(camParams['RightCamK'], camParams['RightCamRT'])
    F = camParams['FMatrix']

    points_3D = []
    # errors = []

    # if len(left_pts.shape) == 1 and len(right_pts.shape) == 1:
    #     return []

    for i in tqdm(range(len(left_pts))):
        # print(left_pts[i])
        u, v = left_pts[i]          # set u, v (x, y in the left image)
        up, vp = right_pts[i]       # set up, vp (x, y in the right image)
        epipolar_line = np.dot(F, np.array([u, v, 1]))  # 不須用到

        # Create matrix for triangulation
        A = np.vstack((
                        u * left_camera_P[2] - left_camera_P[0],
                        v * left_camera_P[2] - left_camera_P[1],
                        up * right_camera_P[2] - right_camera_P[0],
                        vp * right_camera_P[2] - right_camera_P[1]
                     ))

        _, _, vt = np.linalg.svd(A)     # Perform SVD
        X = vt[-1]                      # The 3D point is the last column of V (Vt's last row)
        X /= X[3]                       # Normalize the point
        # error = (sum([x**2 for x in np.dot(A, X)])) ** 0.5
        # errors.append(error)

        X /= 1000   # mm 轉為公尺
        points_3D.append(X[:3])

    return np.array(points_3D, dtype=np.float32)

def transform_coord_system(points_3D, basis_points, origin_indices=(0, 3)):
    i, j = origin_indices
    Pi = basis_points[i]
    Pj = basis_points[j]
    origin = (Pi + Pj) / 2

    # x 軸從中點指向 Pj
    x_axis = Pj - origin
    x_axis /= np.linalg.norm(x_axis)

    # 找一個第三個點來估算 y 軸方向，這裡預設選另一個不是 i 或 j 的點
    k = next(idx for idx in range(len(basis_points)) if idx not in origin_indices)
    y_hint = basis_points[k] - Pi
    y_axis = y_hint - np.dot(y_hint, x_axis) * x_axis  # 投影出去 x 軸的部分
    y_axis /= np.linalg.norm(y_axis)

    # z 軸根據右手定則
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis)

    # 重新定義 y 軸（確保與 x, z 正交）
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    R = np.stack([x_axis, y_axis, z_axis], axis=1)
    transformed = (points_3D - origin) @ R

    return transformed, R

# 簡易 Kalman filter：使用 constant velocity model 進行平滑
def simple_kalman_filter_3d(trajectory, R=25.0, Q=1.0):
    n = len(trajectory)
    dt = 1/225  # 假設等間隔

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

def compute_marker_3d(
        mark_type='mark_o',
        K_left=None, RT_left=None, uv_left=None,
        K_right=None, RT_right=None, uv_right=None,
        C_ball=None, r=0.02
    ):
    """
    支援兩種標記類型：'mark_o'（主標記）與 'mark_x'（球背面）
    若為 'mark_x'，則自動轉換為球體正對面的 'mark_o' 位置（即從球心對稱）

    回傳：
    - P_mark: 3D 座標
    - reproj_error_left/right: 投影誤差
    """

    def get_ray(K, RT, uv):
        uv_h = np.array([uv[0], uv[1], 1.0])
        d_cam = np.linalg.inv(K) @ uv_h
        d_cam /= np.linalg.norm(d_cam)
        R = RT[:3, :3]
        t = RT[:3, 3]
        d_world = R @ d_cam
        d_world /= np.linalg.norm(d_world)
        O_cam = -R.T @ t
        return O_cam, d_world, R, t

    # 選擇一個視角作為主視角（預設左）
    if uv_left is not None and K_left is not None and RT_left is not None:
        O_cam, d_world, R_used, t_used = get_ray(K_left, RT_left, uv_left)
    elif uv_right is not None and K_right is not None and RT_right is not None:
        O_cam, d_world, R_used, t_used = get_ray(K_right, RT_right, uv_right)
    else:
        raise ValueError("請至少提供一組完整的相機內參、外參與影像座標")

    # 射線與球體交點計算
    OC = O_cam - C_ball
    A = np.dot(d_world, d_world)
    B = 2 * np.dot(d_world, OC)
    C = np.dot(OC, OC) - r**2
    discriminant = B**2 - 4 * A * C
    if discriminant < 0:
        return {"P_mark": None}

    t1 = (-B - np.sqrt(discriminant)) / (2 * A)
    t2 = (-B + np.sqrt(discriminant)) / (2 * A)
    P1 = O_cam + t1 * d_world
    P2 = O_cam + t2 * d_world

    # 法向量檢查，選擇面向相機的那一點
    n1 = (P1 - C_ball) / np.linalg.norm(P1 - C_ball)
    n2 = (P2 - C_ball) / np.linalg.norm(P2 - C_ball)
    dot1 = np.dot(n1, -d_world)
    dot2 = np.dot(n2, -d_world)
    P_mark = P1 if dot1 > dot2 else P2

    # 如果是 mark_x，轉換到球的另一面對稱位置
    if mark_type == 'mark_x':
        direction = P_mark - C_ball
        P_mark = C_ball - direction  # 對稱於球心

    result = {"P_mark": P_mark}

    # 投影誤差函式
    def reprojection_error(P, K, RT, uv_gt):
        R = RT[:3, :3]
        t = RT[:3, 3].reshape(3, 1)
        P_cam = R @ P.reshape(3, 1) + t
        P_proj = K @ P_cam
        P_proj /= P_proj[2]
        uv_proj = P_proj[:2].flatten()
        return np.linalg.norm(uv_proj - np.array(uv_gt))

    # 計算左、右影像的投影誤差
    if uv_left is not None and K_left is not None and RT_left is not None:
        result["reproj_error_left"] = reprojection_error(P_mark, K_left, RT_left, uv_left)
    if uv_right is not None and K_right is not None and RT_right is not None:
        result["reproj_error_right"] = reprojection_error(P_mark, K_right, RT_right, uv_right)

    return result

def get_marks_3D(camParams, traj_3D, lmo, rmo, lmx, rmx):
    marks_3D = []
    for i in range(len(traj_3D)):
        C_ball = traj_3D[i]
        if lmo[i] == rmo[i] == lmx[i] == rmx[i] == None:
            marks_3D.append(None)
            continue

        uv_left, uv_right = lmo[i], rmo[i]
        if uv_left or uv_right:
            mark_type = 'mark_o'
        else:
            uv_left, uv_right = lmx[i], rmx[i]
            mark_type = 'mark_x'

        result = compute_marker_3d(
            mark_type=mark_type,
            K_left=camParams['LeftCamK'], RT_left=camParams['LeftCamRT'], uv_left=uv_left,
            K_right=camParams['RightCamK'], RT_right=camParams['RightCamRT'], uv_right=uv_right,
            C_ball=C_ball, r=0.02
        )

        marks_3D.append(result["P_mark"])
    
    marks_3D = np.array([[0, 0, 0] if p is None else p for p in marks_3D])
    return marks_3D

if __name__ == "__main__":
    camParamsPath = "CameraCalibration/STEREO_IMAGES/cvCalibration_result.txt"
    output_folder = 'OUTPUT'
    camParams = read_calibration_file(camParamsPath)
    