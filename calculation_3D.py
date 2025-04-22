import cv2
import numpy as np
import os
from tqdm import tqdm

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

# def get_LR_centers_with_marks(LR_map, all_2D_centers):
#     """
#     根據 all_2D_centers 建立包含球與標記資訊的結構，
#     並分別輸出對應的座標 list 長度一致。
#     valid_LR_centers = [
#                            {
#                                "L": {"ball": (L_ball_center_x, L_ball_center_y), "mark_x": (L_mark_x_center_x, L_mark_x_center_y)}, 
#                                "R": {"ball": (R_ball_center_x, R_ball_center_y)}
#                            }, 

#                            {
#                                "L": {"ball": (L_ball_center_x, L_ball_center_y), "mark_o": (L_mark_o_center_x, L_mark_o_center_y)}, 
#                                "R": {"ball": (L_ball_center_x, L_ball_center_y), "mark_o": (R_mark_o_center_x, R_mark_o_center_y)}
#                            }, 

#                            {
#                                "L": {"ball": (L_ball_center_x, L_ball_center_y)}, 
#                                "R": {"ball": (R_ball_center_x, R_ball_center_y)}
#                            }, 

#                            ...
#                        ]
#     """
#     # 檢查每個 L-R 配對是否都有球，並加入 mark 資訊
#     valid_LR_centers = []

#     for base in sorted(LR_map.keys()):
#         pair = LR_map[base]
#         L_file, R_file = pair["L"], pair["R"]

#         if L_file and R_file:
#             if 0 in all_2D_centers[L_file] and 0 in all_2D_centers[R_file]:
#                 entry = {"L": {}, "R": {}}

#                 # Ball center
#                 entry["L"]["ball"] = all_2D_centers[L_file][0]
#                 entry["R"]["ball"] = all_2D_centers[R_file][0]

#                 # Optional mark_o (key = 1)
#                 entry["L"]["mark_o"] = all_2D_centers[L_file].get(1, None)
#                 entry["R"]["mark_o"] = all_2D_centers[R_file].get(1, None)

#                 # Optional mark_x (key = 2)
#                 entry["L"]["mark_x"] = all_2D_centers[L_file].get(2, None)
#                 entry["R"]["mark_x"] = all_2D_centers[R_file].get(2, None)

#                 valid_LR_centers.append(entry)

#     # 分別組出六個 list
#     left_balls     = [item["L"]["ball"] for item in valid_LR_centers]
#     right_balls    = [item["R"]["ball"] for item in valid_LR_centers]
#     left_mark_o    = [item["L"].get("mark_o", None) for item in valid_LR_centers]
#     right_mark_o   = [item["R"].get("mark_o", None) for item in valid_LR_centers]
#     left_mark_x    = [item["L"].get("mark_x", None) for item in valid_LR_centers]
#     right_mark_x   = [item["R"].get("mark_x", None) for item in valid_LR_centers]

#     return left_balls, right_balls, left_mark_o, right_mark_o, left_mark_x, right_mark_x

def extract_centers(all_2D_centers):
        import re

        # 解析所有的 frame index
        frame_idxs = set()
        pattern = re.compile(r"image-(\d{4})_[LR]\.txt")
        for fname in all_2D_centers:
            m = pattern.match(fname)
            if m:
                frame_idxs.add(int(m.group(1)))
        if not frame_idxs:
            return {}, {}, {}, {}, {}, {}

        min_idx, max_idx = min(frame_idxs), max(frame_idxs)

        # 準備輸出列表
        lb, rb = [], []
        lmo, rmo = [], []
        lmx, rmx = [], []

        # 依序從最小到最大影格號提取
        for idx in range(min_idx, max_idx + 1):
            # 組出左右檔名
            fname_L = f"image-{idx:04d}_L.txt"
            fname_R = f"image-{idx:04d}_R.txt"

            # 取出對應的 dict（若不存在，當作空 dict 處理）
            cent_L = all_2D_centers.get(fname_L, {})
            cent_R = all_2D_centers.get(fname_R, {})

            # 提取 ball(id=0)、mark_o(id=1)、mark_x(id=2)
            lb.append( cent_L.get(0) )
            lmo.append( cent_L.get(1) )
            lmx.append( cent_L.get(2) )

            rb.append( cent_R.get(0) )
            rmo.append( cent_R.get(1) )
            rmx.append( cent_R.get(2) )

        return lb, rb, lmo, rmo, lmx, rmx

def triangulation(left_camera_P, right_camera_P, u, v, up, vp):

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
    X /= 1000   # mm 轉為公尺
    return X[:3]

def myDLT(camParams, left_pts, right_pts):
    left_camera_P = np.dot(camParams['LeftCamK'], camParams['LeftCamRT'])
    right_camera_P = np.dot(camParams['RightCamK'], camParams['RightCamRT'])
    F = camParams['FMatrix']
    points_3D = []

    left_pts = [(np.nan, np.nan) if pt is None else pt for pt in left_pts ]
    right_pts = [(np.nan, np.nan) if pt is None else pt for pt in right_pts ]

    for i in tqdm(range(len(left_pts))):
        if not np.isnan(left_pts[i][0]) and not np.isnan(right_pts[i][0]):
            u, v = left_pts[i]          # set u, v (x, y in the left image)
            up, vp = right_pts[i]       # set up, vp (x, y in the right image)
            X = triangulation(left_camera_P, right_camera_P, u, v, up, vp)
            points_3D.append(X)
        else:
            points_3D.append([np.nan, np.nan, np.nan])

    return np.array(points_3D, dtype=np.float32)

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
        return {"P_mark": np.array([np.nan, np.nan, np.nan])}

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
    left_camera_P = np.dot(camParams['LeftCamK'], camParams['LeftCamRT'])
    right_camera_P = np.dot(camParams['RightCamK'], camParams['RightCamRT'])
    marks_3D = []
    for i in range(len(traj_3D)):
        C_ball = traj_3D[i]
        if lmo[i] == rmo[i] == lmx[i] == rmx[i] == None:            # 左右都沒偵測到標記
            marks_3D.append(np.array([np.nan, np.nan, np.nan]))
            continue
        
        # # 如果兩邊都偵測到標記 直接DLT
        # if lmo[i] and rmo[i]:
        #     u, v = lmo[i]
        #     up, vp = rmo[i]
        #     P_mark = triangulation(left_camera_P, right_camera_P, u, v, up, vp)
        #     marks_3D.append(P_mark)
        #     continue
        # elif lmx[i] and rmx[i]:
        #     u, v = lmx[i]
        #     up, vp = rmx[i]
        #     P_mark = triangulation(left_camera_P, right_camera_P, u, v, up, vp)
        #     direction = P_mark - C_ball
        #     P_mark = C_ball - direction  # 對稱於球心
        #     marks_3D.append(P_mark)
        #     continue
        
        # 如果只有其中一邊偵測到標記 用球面方程式計算
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

    