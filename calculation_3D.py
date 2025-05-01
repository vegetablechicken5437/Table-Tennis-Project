import cv2
import numpy as np
import os
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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
    """ 
    valid_LR_ball_centers = [
                                {"L": (L_ball_center_x, L_ball_center_y), "R": (R_ball_center_x, R_ball_center_y)}, 
                                {"L": (L_ball_center_x, L_ball_center_y), "R": (R_ball_center_x, R_ball_center_y)}, 
                                ...
                            ]
    """
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

def extract_centers(all_2D_centers, total_frames):
    import re

    # 初始化所有 frame 的值為 None
    lb = [None] * total_frames
    rb = [None] * total_frames
    lmo = [None] * total_frames
    rmo = [None] * total_frames
    lmx = [None] * total_frames
    rmx = [None] * total_frames

    # 正則表達式抓 frame index 和左右
    pattern = re.compile(r"image-(\d{4})_([LR])\.txt")

    for fname, center_dict in all_2D_centers.items():
        m = pattern.match(fname)
        if not m:
            continue
        idx = int(m.group(1))
        side = m.group(2)

        if idx >= total_frames:
            continue  # 忽略超出範圍的 frame

        if side == "L":
            lb[idx] = center_dict.get(0)
            lmo[idx] = center_dict.get(1)
            lmx[idx] = center_dict.get(2)
        elif side == "R":
            rb[idx] = center_dict.get(0)
            rmo[idx] = center_dict.get(1)
            rmx[idx] = center_dict.get(2)

    return lb, rb, lmo, rmo, lmx, rmx

def triangulation(left_camera_P, right_camera_P, u, v, up, vp, to_meter=False):

    # epipolar_line = np.dot(F, np.array([u, v, 1])) 

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
    if to_meter:
        X /= 1000   # mm 轉為公尺
    return X[:3]

# 投影誤差函式
def reprojection_error(P, K, RT, uv_gt):
    P_proj = np.dot(K, RT)
    P = np.array([P[0], P[1], P[2], 1])

    uv_proj = np.dot(P_proj, P)
    uv_proj /= uv_proj[-1]
    uv_proj = uv_proj[:2]

    return uv_proj - uv_gt

def myDLT(camParams, left_pts, right_pts):
    left_camera_P = np.dot(camParams['LeftCamK'], camParams['LeftCamRT'])
    right_camera_P = np.dot(camParams['RightCamK'], camParams['RightCamRT'])
    F = camParams['FMatrix']
    points_3D = []

    left_pts = [(np.nan, np.nan) if pt is None else pt for pt in left_pts ]
    right_pts = [(np.nan, np.nan) if pt is None else pt for pt in right_pts ]

    reproj_errors_L = []
    reproj_errors_R = []

    for i in tqdm(range(len(left_pts))):
        if np.isnan(left_pts[i][0]) or np.isnan(right_pts[i][0]):
            points_3D.append([np.nan, np.nan, np.nan])
            reproj_errors_L.append(np.nan)
            reproj_errors_R.append(np.nan)
        else:
            u, v = left_pts[i]          # set u, v (x, y in the left image)
            up, vp = right_pts[i]       # set up, vp (x, y in the right image)
            X = triangulation(left_camera_P, right_camera_P, u, v, up, vp)
            points_3D.append(X)

            # print(left_pts[i])
            # print(right_pts[i])

            error_L = reprojection_error(X, camParams['LeftCamK'], camParams['LeftCamRT'], (u, v))
            error_R = reprojection_error(X, camParams['RightCamK'], camParams['RightCamRT'], (up, vp))

            reproj_errors_L.append(np.linalg.norm(error_L))
            reproj_errors_R.append(np.linalg.norm(error_R))

            # print(f"Frame {i+1}")
            # print("Reprojection Error Left: ", error_L)
            # print("Reprojection Error Right: ", error_R)

    return np.array(points_3D, dtype=np.float32), reproj_errors_L, reproj_errors_R

def transform_coord_system(points_3D, basis_points, origin_indices=(0, 3)):
    i, j = origin_indices
    Pi = basis_points[i]
    Pj = basis_points[j]
    origin = Pi
    # origin = (Pi + Pj) / 2

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


def mark_x_to_mark_o(P_mark, C_ball):
    # 如果是 mark_x，轉換到球的另一面對稱位置
    direction = P_mark - C_ball
    P_mark = C_ball - direction  # 對稱於球心
    return P_mark


def estimate_marker_3d_on_sphere(K_L, K_R, RT_L, RT_R, uv_L, uv_R, center, radius):
    """
    根據左右相機的 K、RT、2D 座標與球心、半徑，估算標記的 3D 座標
    """

    def get_camera_center_and_rotation(RT):
        R = RT[:3, :3]
        t = RT[:3, 3]
        C = -R.T @ t
        return C, R

    def pixel_to_ray(K, uv, R, C):
        uv_h = np.array([uv[0], uv[1], 1.0])
        d_cam = np.linalg.inv(K) @ uv_h
        d_world = R.T @ d_cam
        d_world /= np.linalg.norm(d_world)
        return C, d_world

    def intersect_ray_sphere(C, d, center, radius):
        oc = C - center
        a = np.dot(d, d)
        b = 2.0 * np.dot(oc, d)
        c = np.dot(oc, oc) - radius**2
        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            return None
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2*a)
        t2 = (-b + sqrt_disc) / (2*a)
        p1 = C + t1 * d
        p2 = C + t2 * d
        return p1, p2

    def select_closest(points, center):
        return min(points, key=lambda p: np.linalg.norm(p - center))

    # 左右相機資訊
    C_L, R_L = get_camera_center_and_rotation(RT_L)
    C_R, R_R = get_camera_center_and_rotation(RT_R)

    # 左右相機視線
    origin_L, dir_L = pixel_to_ray(K_L, uv_L, R_L, C_L)
    origin_R, dir_R = pixel_to_ray(K_R, uv_R, R_R, C_R)

    # 與球面交點
    P_L = intersect_ray_sphere(origin_L, dir_L, center, radius)
    P_R = intersect_ray_sphere(origin_R, dir_R, center, radius)
    if P_L is None or P_R is None:
        return np.array([np.nan, np.nan, np.nan])

    P_L_sel = select_closest(P_L, center)
    P_R_sel = select_closest(P_R, center)

    P_est_raw = (P_L_sel + P_R_sel) / 2  # 線性平均
    P_est = center + radius * (P_est_raw - center) / np.linalg.norm(P_est_raw - center)  # 強制貼合球面

    return P_est


def get_marks_3D(camParams, traj_3D, lmo, rmo, lmx, rmx):

    K_L, K_R = camParams['LeftCamK'], camParams['RightCamK']
    RT_L, RT_R = camParams['LeftCamRT'], camParams['RightCamRT']

    marks_3D = []
    for i in range(len(traj_3D)):

        C_ball = traj_3D[i]
        
        # 如果只有其中一邊偵測到標記 用球面方程式計算
        if lmo[i] and rmo[i]:
            uv_L, uv_R = lmo[i], rmo[i]
            mark_type = 'mark_o'
        elif lmx[i] and rmx[i]:
            uv_L, uv_R = lmx[i], rmx[i]
            mark_type = 'mark_x'
        else:
            marks_3D.append(np.array([np.nan, np.nan, np.nan]))
            continue

        P_mark = estimate_marker_3d_on_sphere(K_L, K_R, RT_L, RT_R, uv_L, uv_R, C_ball, radius=20)

        if mark_type == 'mark_x':
            P_mark = mark_x_to_mark_o(P_mark, C_ball)

        marks_3D.append(P_mark)
    
    return np.array(marks_3D)

if __name__ == "__main__":

    all_2D_centers = {
        "image-0000_L.txt": {0:(10,20), 1:(15,25)},
        "image-0001_R.txt": {0:(30,40), 2:(35,45)},
        "image-0002_L.txt": {0:(50,60)},
        "image-0002_R.txt": {0:(70,80)},
    }
    lb, rb, lmo, rmo, lmx, rmx = extract_centers(all_2D_centers)

    print("lb :", lb)
    print("rb :", rb)
    print("lmo:", lmo)
    print("rmo:", rmo)
    print("lmx:", lmx)
    print("rmx:", rmx)

    