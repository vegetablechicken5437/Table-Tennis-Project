import cv2
import numpy as np
import os
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from visualize_functions import plot_reprojection_error

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

            error_L = reprojection_error(X, camParams['LeftCamK'], camParams['LeftCamRT'], (u, v))
            error_R = reprojection_error(X, camParams['RightCamK'], camParams['RightCamRT'], (up, vp))

            if np.linalg.norm(error_L) < 10 and np.linalg.norm(error_R) < 10:
                points_3D.append(X)
                reproj_errors_L.append(np.linalg.norm(error_L))
                reproj_errors_R.append(np.linalg.norm(error_R))
            else:
                points_3D.append([np.nan, np.nan, np.nan])

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

# def estimate_marker_3d_on_sphere(K_L, K_R, RT_L, RT_R, uv_L, uv_R, center, radius):
#     """
#     根據左右相機的 K、RT、2D 座標與球心、半徑，估算標記的 3D 座標
#     """

#     def get_camera_center_and_rotation(RT):
#         R = RT[:3, :3]
#         t = RT[:3, 3]
#         C = -R.T @ t
#         return C, R

#     def pixel_to_ray(K, uv, R):
#         uv_h = np.array([uv[0], uv[1], 1.0])
#         d_cam = np.linalg.inv(K) @ uv_h
#         d_world = R.T @ d_cam
#         d_world /= np.linalg.norm(d_world)
#         return d_world

#     def intersect_ray_sphere(C, d, center, radius):
#         oc = C - center
#         a = np.dot(d, d)
#         b = 2.0 * np.dot(oc, d)
#         c = np.dot(oc, oc) - radius**2
#         discriminant = b**2 - 4*a*c
#         if discriminant < 0:
#             return None
#         sqrt_disc = np.sqrt(discriminant)
#         t1 = (-b - sqrt_disc) / (2*a)
#         t2 = (-b + sqrt_disc) / (2*a)
#         p1 = C + t1 * d
#         p2 = C + t2 * d
#         return p1, p2

#     def select_closest(points, center):
#         return min(points, key=lambda p: np.linalg.norm(p - center))

#     # 左右相機資訊
#     C_L, R_L = get_camera_center_and_rotation(RT_L)
#     C_R, R_R = get_camera_center_and_rotation(RT_R)

#     # 左右相機視線
#     dir_L = pixel_to_ray(K_L, uv_L, R_L)
#     dir_R = pixel_to_ray(K_R, uv_R, R_R)

#     # 與球面交點
#     P_L = intersect_ray_sphere(C_L, dir_L, center, radius)
#     P_R = intersect_ray_sphere(C_R, dir_R, center, radius)
#     if P_L is None or P_R is None:
#         return np.array([np.nan, np.nan, np.nan])

#     P_L_sel = select_closest(P_L, center)
#     P_R_sel = select_closest(P_R, center)

#     P_est_raw = (P_L_sel + P_R_sel) / 2  # 線性平均
#     P_est = center + radius * (P_est_raw - center) / np.linalg.norm(P_est_raw - center)  # 強制貼合球面

#     return P_est

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

def visualize_ray_sphere_intersection_with_estimation(K_L, K_R, RT_L, RT_R, uv_L, uv_R, center, radius,
                                                  output_dir='my_results', filename='marker_debug_case1.html'):
    """
    視覺化左右相機射線與球面交點，並顯示估算的P_est點
    """

    def get_camera_center_and_rotation(RT):
        R = RT[:3, :3]
        t = RT[:3, 3]
        C = -R.T @ t
        return C, R

    def pixel_to_ray(K, uv, R):
        uv_h = np.array([uv[0], uv[1], 1.0])
        d_cam = np.linalg.inv(K) @ uv_h
        d_world = R.T @ d_cam
        return d_world / np.linalg.norm(d_world)

    def intersect_ray_sphere(C, d, center, radius):
        """
        計算射線 C + t*d 與球體的第一個正向交點（離相機最近，且 t > 0）
        """
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

        valid_ts = [t for t in [t1, t2] if t > 0]
        if not valid_ts:
            return None

        t_nearest = min(valid_ts)
        return (C + t_nearest * d,)

    def select_closest(points, center):
        return min(points, key=lambda p: np.linalg.norm(p - center))

    # 相機位置與方向
    C_L, R_L = get_camera_center_and_rotation(RT_L)
    C_R, R_R = get_camera_center_and_rotation(RT_R)
    d_L = pixel_to_ray(K_L, uv_L, R_L)
    d_R = pixel_to_ray(K_R, uv_R, R_R)

    # 射線與球面交點
    P_L_pair = intersect_ray_sphere(C_L, d_L, center, radius)
    P_R_pair = intersect_ray_sphere(C_R, d_R, center, radius)

    fig = go.Figure()

    # 畫球面
    u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
    x = center[0] + radius * np.cos(u) * np.sin(v)
    y = center[1] + radius * np.sin(u) * np.sin(v)
    z = center[2] + radius * np.cos(v)
    fig.add_trace(go.Surface(x=x, y=y, z=z, opacity=0.3, showscale=False, colorscale='Blues'))

    # 相機位置
    fig.add_trace(go.Scatter3d(x=[C_L[0]], y=[C_L[1]], z=[C_L[2]], mode='markers', marker=dict(size=6, color='red'), name='Cam L'))
    fig.add_trace(go.Scatter3d(x=[C_R[0]], y=[C_R[1]], z=[C_R[2]], mode='markers', marker=dict(size=6, color='green'), name='Cam R'))

    # 射線
    fig.add_trace(go.Scatter3d(x=[C_L[0], C_L[0] + d_L[0]*5000],
                               y=[C_L[1], C_L[1] + d_L[1]*5000],
                               z=[C_L[2], C_L[2] + d_L[2]*5000],
                               mode='lines', line=dict(color='red'), name='Ray L'))

    fig.add_trace(go.Scatter3d(x=[C_R[0], C_R[0] + d_R[0]*5000],
                               y=[C_R[1], C_R[1] + d_R[1]*5000],
                               z=[C_R[2], C_R[2] + d_R[2]*5000],
                               mode='lines', line=dict(color='green'), name='Ray R'))

    # 交點與估算點
    if P_L_pair is not None and P_R_pair is not None:
        P_L_sel = select_closest(P_L_pair, center)
        P_R_sel = select_closest(P_R_pair, center)

        fig.add_trace(go.Scatter3d(x=[P_L_sel[0]], y=[P_L_sel[1]], z=[P_L_sel[2]],
                                   mode='markers', marker=dict(size=4, color='red'), name='Intersect L'))
        fig.add_trace(go.Scatter3d(x=[P_R_sel[0]], y=[P_R_sel[1]], z=[P_R_sel[2]],
                                   mode='markers', marker=dict(size=4, color='green'), name='Intersect R'))
        
        # 左右相機視線方向箭頭（單位方向 * 長度）
        vis_len = 300  # 可調整視線箭頭的長度

        fig.add_trace(go.Scatter3d(
            x=[C_L[0], C_L[0] + d_L[0]*vis_len],
            y=[C_L[1], C_L[1] + d_L[1]*vis_len],
            z=[C_L[2], C_L[2] + d_L[2]*vis_len],
            mode='lines+markers',
            line=dict(color='darkred', dash='dot', width=3),
            marker=dict(size=2),
            name='L: viewing direction'
        ))

        fig.add_trace(go.Scatter3d(
            x=[C_R[0], C_R[0] + d_R[0]*vis_len],
            y=[C_R[1], C_R[1] + d_R[1]*vis_len],
            z=[C_R[2], C_R[2] + d_R[2]*vis_len],
            mode='lines+markers',
            line=dict(color='darkgreen', dash='dot', width=3),
            marker=dict(size=2),
            name='R: viewing direction'
        ))

        # 預估點（貼合球面）
        P_est_raw = (P_L_sel + P_R_sel) / 2
        P_est = center + radius * (P_est_raw - center) / np.linalg.norm(P_est_raw - center)

        fig.add_trace(go.Scatter3d(x=[P_est[0]], y=[P_est[1]], z=[P_est[2]],
                                   mode='markers', marker=dict(size=6, color='blue'), name='P_est'))
    else:
        P_est = np.array([np.nan, np.nan, np.nan])

    fig.update_layout(scene=dict(aspectmode='data'),
                      title='Ray-Sphere Intersection and P_est Visualization')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    pio.write_html(fig, file=output_path, auto_open=False)

    return P_est


def get_marks_3D(camParams, traj_3D, lmo, rmo, lmx, rmx, output_dir=None):

    K_L, K_R = camParams['LeftCamK'], camParams['RightCamK']
    RT_L, RT_R = camParams['LeftCamRT'], camParams['RightCamRT']

    marks_3D, reproj_errors_L, reproj_errors_R = [], [], []

    print(f"🚀 計算並視覺化標記3D座標...")
    for i in tqdm(range(len(traj_3D))):
        C_ball = traj_3D[i]
        # 如果兩邊都偵測到標記，計算左右相機連接標記的射線 射出後 與球面的交點
        if lmo[i] and rmo[i]:
            uv_L, uv_R = lmo[i], rmo[i]
            mark_type = 'mark_o'
        elif lmx[i] and rmx[i]:
            uv_L, uv_R = lmx[i], rmx[i]
            mark_type = 'mark_x'
        else:   
            marks_3D.append(np.array([np.nan, np.nan, np.nan]))
            reproj_errors_L.append(np.nan)
            reproj_errors_R.append(np.nan)
            continue

        P_mark = visualize_ray_sphere_intersection_with_estimation(K_L, K_R, RT_L, RT_R, uv_L, uv_R, C_ball, radius=20,
                                                                   output_dir=output_dir, 
                                                                   filename=f'frame_{i+1}.html')

        if np.isnan(P_mark[0]):
            marks_3D.append(P_mark)
            reproj_errors_L.append(np.nan)
            reproj_errors_R.append(np.nan)
            continue

        error_L = reprojection_error(P_mark, K_L, RT_L, uv_L)
        error_R = reprojection_error(P_mark, K_R, RT_R, uv_R)
        reproj_errors_L.append(np.linalg.norm(error_L))
        reproj_errors_R.append(np.linalg.norm(error_R))

        if mark_type == 'mark_x':
            P_mark = mark_x_to_mark_o(P_mark, C_ball)

        marks_3D.append(P_mark)
    
    print(f"✅ [相機與標記相對位置圖] 已輸出至：{output_dir}")
    return np.array(marks_3D), reproj_errors_L, reproj_errors_R

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

    