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

def draw_epipolar_line(image, a, b, c, color=(0, 255, 0), thickness=1):
    height, width = image.shape[:2]
    
    # 左右影像的寬度區間
    left_width = width // 2
    right_start = left_width
    right_end = width

    points = []

    # 計算與右半部分 y = 0 和 y = height-1 的交點
    if b != 0:  # 確保不會除以零
        x0 = int(-c / a) if a != 0 else None  # 與 y = 0 的交點
        x1 = int(-(c + b * (height - 1)) / a) if a != 0 else None  # 與 y = height-1 的交點
        if x0 is not None and right_start <= x0 < right_end:
            points.append((x0, 0))
        if x1 is not None and right_start <= x1 < right_end:
            points.append((x1, height - 1))

    # 計算與右半部分 x = W/2 和 x = W 的交點
    if a != 0:  # 確保不會除以零
        y0 = int(-(c + a * right_start) / b) if b != 0 else None  # 與 x = W/2 的交點
        y1 = int(-(c + a * (right_end - 1)) / b) if b != 0 else None  # 與 x = W 的交點
        if y0 is not None and 0 <= y0 < height:
            points.append((right_start, y0))
        if y1 is not None and 0 <= y1 < height:
            points.append((right_end - 1, y1))

    # 把點移到右半區域內，並畫出對極線
    if len(points) == 2:
        cv2.line(image, points[0], points[1], color, thickness)
    
    return image

def myDLT(camParams, left_pts, right_pts):
    left_camera_P = np.dot(camParams['LeftCamK'], camParams['LeftCamRT'])
    right_camera_P = np.dot(camParams['RightCamK'], camParams['RightCamRT'])
    F = camParams['FMatrix']

    points_3D = []
    errors = []

    if len(left_pts.shape) == 1 and len(right_pts.shape) == 1:
        return []

    for i in tqdm(range(len(left_pts))):
        # print(left_pts[i])
        u, v = left_pts[i]          # set u, v (x, y in the left image)
        up, vp = right_pts[i]       # set up, vp (x, y in the right image)
        epipolar_line = np.dot(F, np.array([u, v, 1]))  # 不須用到

        # Create matrix for triangulation
        A = np.vstack(( u * left_camera_P[2] - left_camera_P[0],
                        v * left_camera_P[2] - left_camera_P[1],
                        up * right_camera_P[2] - right_camera_P[0],
                        vp * right_camera_P[2] - right_camera_P[1]))

        _, _, vt = np.linalg.svd(A)     # Perform SVD
        X = vt[-1]                      # The 3D point is the last column of V (Vt's last row)
        X /= X[3]                       # Normalize the point
        error = (sum([x**2 for x in np.dot(A, X)])) ** 0.5
        errors.append(error)
        X /= 1000
        points_3D.append(X[:3])

    return np.array(points_3D, dtype=np.float32)

def normalize(v):
    """ 將向量標準化為單位向量 """
    return v / np.linalg.norm(v)

def define_coordinate_system(P1, P2, P3, P4):
    # 計算 +y 軸向量
    y_axis = P2 - P1
    y_axis = normalize(y_axis)

    # 計算 +x 軸向量
    x_axis = P4 - P1
    x_axis = normalize(x_axis)

    # 計算 +z 軸為 x 軸與 y 軸的叉積
    z_axis = np.cross(x_axis, y_axis)
    z_axis = normalize(z_axis)

    # 再次計算正交的 x 軸（正交化後的 x 軸）
    x_axis_orthogonal = np.cross(y_axis, z_axis)
    x_axis_orthogonal = normalize(x_axis_orthogonal)

    return x_axis_orthogonal, y_axis, z_axis

def transform_point(P, P1, R):
    """ 將 3D 點從舊的座標系轉換到新的座標系 """
    # 計算相對於 P1 的點
    relative_P = P - P1
    # 使用旋轉矩陣進行變換
    new_P = np.dot(R, relative_P)
    return new_P

def changeCoordSys(corners_3D, points_3D, point_3D_path):

    P1, P2, P3, P4 = corners_3D
    x_axis, y_axis, z_axis = define_coordinate_system(P1, P2, P3, P4)

    # 構建旋轉矩陣 R
    R = np.vstack([x_axis, y_axis, z_axis])  # 旋轉矩陣，行向量為新的 x, y, z 軸

    # 轉換這些點到新的座標系
    transformed_points = []
    for point in points_3D:
        new_point = transform_point(point, P1, R)
        transformed_points.append(new_point)

    np.savetxt(point_3D_path, transformed_points, fmt="%.4f")
    print(f'✅ 3D points saved in {point_3D_path}')

if __name__ == "__main__":
    camParamsPath = "CameraCalibration/STEREO_IMAGES/cvCalibration_result.txt"
    output_folder = 'OUTPUT'
    camParams = read_calibration_file(camParamsPath)
    