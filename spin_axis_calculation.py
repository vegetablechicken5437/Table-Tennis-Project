import numpy as np

def ransac_fit_plane(offsets, iterations=100, threshold=0.1):
    offsets_clean = offsets[~np.isnan(offsets).any(axis=1)]
    if len(offsets_clean) < 3:
        return {'normal': np.array([np.nan]*3), 'x_axis': None, 'y_axis': None}, offsets

    best_inliers = []
    best_normal = None

    for _ in range(iterations):
        # 隨機取 3 點
        sample = offsets_clean[np.random.choice(len(offsets_clean), 3, replace=False)]
        v1 = sample[1] - sample[0]
        v2 = sample[2] - sample[0]
        normal = np.cross(v1, v2)
        if np.linalg.norm(normal) == 0:
            continue
        normal = normal / np.linalg.norm(normal)
        d = -np.dot(normal, sample[0])

        # 計算所有點到平面距離
        distances = np.abs(offsets_clean @ normal + d)

        # 找內點
        inliers = offsets_clean[distances < threshold]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_normal = normal

    if len(best_inliers) < 3:
        return {'normal': np.array([np.nan]*3), 'x_axis': None, 'y_axis': None}, offsets

    # 對內點做 SVD 擬合平面
    centroid = best_inliers.mean(axis=0)
    u, s, vh = np.linalg.svd(best_inliers - centroid)
    normal = vh[-1]
    x_axis = vh[0]
    y_axis = vh[1]
    plane = {'normal': normal, 'x_axis': x_axis, 'y_axis': y_axis}

    # 標記 filtered_offsets（不是內點就設為 nan）
    filtered_offsets = []
    for o in offsets:
        if np.isnan(o).any():
            filtered_offsets.append(np.array([np.nan]*3))
        elif np.any(np.all(np.isclose(o, best_inliers, atol=1e-6), axis=1)):
            filtered_offsets.append(o)
        else:
            filtered_offsets.append(np.array([np.nan]*3))

    return plane, filtered_offsets

def fit_plane_with_prior(offsets, prior=np.array([1.0, 0.0, 0.0]), lam=10.0):
    """
    擬合平面，使其法向量接近 prior（預設為 x 軸），根據 offsets 微調
    """
    offsets_clean = offsets[~np.isnan(offsets).any(axis=1)]
    if len(offsets_clean) < 3:
        return {'normal': np.array([np.nan]*3), 'x_axis': None, 'y_axis': None}, offsets

    # 中心化資料
    centroid = offsets_clean.mean(axis=0)
    X = offsets_clean - centroid

    # 資料協方差矩陣
    C = X.T @ X

    # 加入先驗方向的懲罰
    P = lam * (np.eye(3) - np.outer(prior, prior))  # 懲罰偏離 prior 的方向
    A = C + P

    # 求 A 的最小特徵值對應向量（最佳法向量）
    eigvals, eigvecs = np.linalg.eigh(A)
    normal = eigvecs[:, np.argmin(eigvals)]
    normal = normal / np.linalg.norm(normal)

    # 建立正交基底
    arbitrary = np.array([0, 1, 0]) if abs(normal[0]) > 0.9 else np.array([1, 0, 0])
    x_axis = np.cross(normal, arbitrary)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(normal, x_axis)

    plane = {'normal': normal, 'x_axis': x_axis, 'y_axis': y_axis}

    # 建立 filtered_offsets（無變化，標記有效資料）
    filtered_offsets = []
    for o in offsets:
        if np.isnan(o).any():
            filtered_offsets.append(np.array([np.nan]*3))
        else:
            filtered_offsets.append(o)

    return plane, np.array(filtered_offsets)


if __name__ == "__main__":

    traj_3D = np.loadtxt('OUTPUT/0408/20250408_193842/traj_3D.txt')
    marks_3D = np.loadtxt('OUTPUT/0408/20250408_193842/marks_3D.txt')
    marks_3D = np.loadtxt(r"C:\Users\jason\Desktop\TableTennisProject\OUTPUT\0415\20250415_193043\marks_3D_transformed.txt")

    print(marks_3D)
    print(marks_3D[~np.isnan(marks_3D).any(axis=1)])

