import numpy as np

# def fit_offset_plane(offsets, angle_thres=45):  # angle_thres in degrees
#     offsets_clean = offsets[~np.isnan(offsets).any(axis=1)]  # 過濾 np.nan

#     if len(offsets_clean) < 3:
#         print("❗需要至少三個有效點才能擬合平面")
#         plane = {'normal': np.array([np.nan, np.nan, np.nan]), 'x_axis': None, 'y_axis': None}
#         return plane, offsets

#     # 擬合平面
#     centroid = offsets_clean.mean(axis=0)
#     u, s, vh = np.linalg.svd(offsets_clean - centroid)
#     normal = vh[-1]
#     x_axis = vh[0]
#     y_axis = vh[1]
#     plane = {'normal': normal, 'x_axis': x_axis, 'y_axis': y_axis}

#     filtered_offsets = []
#     for o in offsets:
#         if np.isnan(o).any():
#             filtered_offsets.append(np.array([np.nan, np.nan, np.nan]))
#             continue

#         cos_theta = np.clip(np.dot(o, normal) / (np.linalg.norm(o) * np.linalg.norm(normal)), -1.0, 1.0)
#         theta = np.arccos(cos_theta)
#         theta = theta * 360 / (2 * np.pi)

#         if theta > angle_thres and theta < 180 - angle_thres:
#             filtered_offsets.append(o)
#         else:
#             filtered_offsets.append(np.array([np.nan, np.nan, np.nan]))

#     return plane, filtered_offsets

import numpy as np

def ransac_fit_plane(offsets, iterations=100, threshold=0.1):
    offsets_clean = offsets[~np.isnan(offsets).any(axis=1)]
    if len(offsets_clean) < 3:
        return {'normal': np.array([np.nan]*3), 'x_axis': None, 'y_axis': None}, offsets
    
    # # 篩除 nan 及加入對面點
    # offsets_opposite = -offsets_clean
    # offsets_extended = np.vstack((offsets_clean, offsets_opposite))
    # offsets_clean = offsets_extended

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



def compute_spin_axis_directions(px_list, py_list, pz_list, aero_params, dt, original_trajs):
    rps_list = []
    g, m, rho, A, r, Cd, Cm = aero_params.values()

    for px, py, pz, traj in zip(px_list, py_list, pz_list, original_trajs):
        t = np.arange(len(traj)) * dt
        t0 = t[len(t) // 2]

        v0 = np.array([px.deriv(1)(t0), py.deriv(1)(t0), pz.deriv(1)(t0)])
        a0 = np.array([px.deriv(2)(t0), py.deriv(2)(t0), pz.deriv(2)(t0)])
        vnorm = np.linalg.norm(v0)
        Fd = -0.5 * Cd * rho * A * vnorm * v0
        Fnet = m * a0 - m * g - Fd

        omega_vec = np.cross(Fnet, v0) / (vnorm ** 2)
        omega_vec *= 3 / (4 * Cm * np.pi * r**3 * rho)
        omega_vec = omega_vec / np.linalg.norm(omega_vec) * 0.3  # normalize and scale

        rps_list.append((omega_vec, t0))  # 也可以只 append omega_vec

    return rps_list


if __name__ == "__main__":

    traj_3D = np.loadtxt('OUTPUT/0408/20250408_193842/traj_3D.txt')
    marks_3D = np.loadtxt('OUTPUT/0408/20250408_193842/marks_3D.txt')
    marks_3D = np.loadtxt(r"C:\Users\jason\Desktop\TableTennisProject\OUTPUT\0415\20250415_193043\marks_3D_transformed.txt")

    print(marks_3D)
    print(marks_3D[~np.isnan(marks_3D).any(axis=1)])

