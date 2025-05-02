import numpy as np

# def fit_offset_plane(offsets, r=20, thres=0.8):

#     offsets_clean = offsets[~np.isnan(offsets).any(axis=1)]     # 過濾 np.nan

#     if len(offsets_clean) < 3:
#         print("❗需要至少三個有效點才能擬合平面")
#         return np.array([np.nan, np.nan, np.nan]), offsets

#     # 擬合平面
#     centroid = offsets_clean.mean(axis=0)
#     u, s, vh = np.linalg.svd(offsets_clean - centroid)
#     normal = vh[-1]
#     x_axis = vh[0]
#     y_axis = vh[1]
#     plane = {'normal':normal, 'x_axis':x_axis, 'y_axis':y_axis}

#     filtered_offsets = []
#     for o in offsets:
#         if o[0] == np.nan:
#             filtered_offsets.append(np.array([np.nan, np.nan, np.nan]))
#             continue
#         vec = np.array(o) - centroid
#         proj_x = np.dot(vec, x_axis)
#         proj_y = np.dot(vec, y_axis)
#         proj_len = np.sqrt(proj_x**2 + proj_y**2)
#         if proj_len < r * thres:
#             filtered_offsets.append(np.array([np.nan, np.nan, np.nan]))
#         else:
#             filtered_offsets.append(np.array(o))

#     return plane, filtered_offsets

import numpy as np

def fit_offset_plane(offsets, angle_thres=45):  # angle_thres in degrees
    offsets_clean = offsets[~np.isnan(offsets).any(axis=1)]  # 過濾 np.nan

    if len(offsets_clean) < 3:
        print("❗需要至少三個有效點才能擬合平面")
        return np.array([np.nan, np.nan, np.nan]), offsets

    # 擬合平面
    centroid = offsets_clean.mean(axis=0)
    u, s, vh = np.linalg.svd(offsets_clean - centroid)
    normal = vh[-1]
    x_axis = vh[0]
    y_axis = vh[1]
    plane = {'normal': normal, 'x_axis': x_axis, 'y_axis': y_axis}

    angle_cos_thres = np.cos(np.deg2rad(angle_thres))

    filtered_offsets = []
    for o in offsets:
        if np.isnan(o).any():
            filtered_offsets.append(np.array([np.nan, np.nan, np.nan]))
            continue

        cos_theta = np.abs(np.dot(o, normal))
        # cos_theta = np.abs(np.dot(vec / vec_norm, normal))
        if cos_theta < angle_cos_thres:  # 夾角 > angle_thres
            filtered_offsets.append(np.array([np.nan, np.nan, np.nan]))
        else:
            filtered_offsets.append(o)

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

