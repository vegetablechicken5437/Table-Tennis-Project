import numpy as np

# 內參矩陣
K_L = np.array([
    [1579.796733, 0.000000, 758.808851],
    [0.000000, 1582.535320, 571.716867],
    [0.000000, 0.000000, 1.000000]
])

K_R = np.array([
    [1590.733228, 0.000000, 711.296231],
    [0.000000, 1594.723884, 577.111844],
    [0.000000, 0.000000, 1.000000]
])

# 外參矩陣
RT_L = np.eye(4)

RT_R = np.array([
    [0.906328, 0.002155, 0.422570, -711.662615],
    [-0.001459, 0.999997, -0.001969, 9.478345],
    [-0.422573, 0.001168, 0.906328, 187.220871],
    [0, 0, 0, 1]
])

# 球心與半徑（單位：mm）
center = np.array([422.01, 188.94, 1166.6])
radius = 20.0

# 標記點影像座標
uv_L = np.array([1327, 810])
uv_R = np.array([945, 865])

# 相機中心與旋轉矩陣
def get_camera_center_and_rotation(RT):
    R = RT[:3, :3]
    t = RT[:3, 3]
    C = -R.T @ t
    return C, R

C_L, R_L = get_camera_center_and_rotation(RT_L)
C_R, R_R = get_camera_center_and_rotation(RT_R)

# 將 2D 像素座標轉換為 3D 視線
def pixel_to_ray(K, uv, R, C):
    uv_h = np.array([uv[0], uv[1], 1.0])
    d_cam = np.linalg.inv(K) @ uv_h
    d_world = R.T @ d_cam
    d_world /= np.linalg.norm(d_world)
    return C, d_world

origin_L, dir_L = pixel_to_ray(K_L, uv_L, R_L, C_L)
origin_R, dir_R = pixel_to_ray(K_R, uv_R, R_R, C_R)

# 與球面交點
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

P_L = intersect_ray_sphere(origin_L, dir_L, center, radius)
P_R = intersect_ray_sphere(origin_R, dir_R, center, radius)
P_L_sel = select_closest(P_L, center)
P_R_sel = select_closest(P_R, center)

# 平均為估算標記點位置
P_est = (P_L_sel + P_R_sel) / 2



# import plotly.graph_objs as go
# import numpy as np

# # 假設你已經有以下資料（請代入你的變數）
# center = np.array([422.01, 188.94, 1166.6])
# radius = 20.0
# P_est = np.array([417.691591, 178.447268, 1151.376313])  # 估算標記
# C_L = np.array([0, 0, 0])
# C_R = np.array([711.662615, -9.478345, -187.220871])
# P_L_sel = np.array([418.5, 179.2, 1151.9])
# P_R_sel = np.array([416.8, 177.7, 1150.9])

# # 建立球面
# phi, theta = np.mgrid[0:np.pi:30j, 0:2*np.pi:30j]
# x_sphere = center[0] + radius * np.sin(phi) * np.cos(theta)
# y_sphere = center[1] + radius * np.sin(phi) * np.sin(theta)
# z_sphere = center[2] + radius * np.cos(phi)

# fig = go.Figure()

# # 球體
# fig.add_trace(go.Surface(
#     x=x_sphere, y=y_sphere, z=z_sphere,
#     opacity=0.3, colorscale='Blues', showscale=False
# ))

# # 左右相機位置
# fig.add_trace(go.Scatter3d(x=[C_L[0]], y=[C_L[1]], z=[C_L[2]],
#                            mode='markers+text', marker=dict(size=6, color='red'),
#                            text=['Left Cam'], name='Left Camera'))
# fig.add_trace(go.Scatter3d(x=[C_R[0]], y=[C_R[1]], z=[C_R[2]],
#                            mode='markers+text', marker=dict(size=6, color='green'),
#                            text=['Right Cam'], name='Right Camera'))

# # 標記點與球心
# fig.add_trace(go.Scatter3d(x=[P_est[0]], y=[P_est[1]], z=[P_est[2]],
#                            mode='markers+text', marker=dict(size=6, color='orange'),
#                            text=['Est Marker'], name='Estimated Marker'))

# fig.add_trace(go.Scatter3d(x=[center[0]], y=[center[1]], z=[center[2]],
#                            mode='markers+text', marker=dict(size=6, color='blue'),
#                            text=['Ball Center'], name='Ball Center'))

# # 視線
# fig.add_trace(go.Scatter3d(x=[C_L[0], P_L_sel[0]], y=[C_L[1], P_L_sel[1]], z=[C_L[2], P_L_sel[2]],
#                            mode='lines', line=dict(color='red'), name='Ray L'))

# fig.add_trace(go.Scatter3d(x=[C_R[0], P_R_sel[0]], y=[C_R[1], P_R_sel[1]], z=[C_R[2], P_R_sel[2]],
#                            mode='lines', line=dict(color='green'), name='Ray R'))

# fig.update_layout(
#     scene=dict(aspectmode='data'),
#     title="3D Visualization of Marker on Sphere",
#     margin=dict(l=0, r=0, b=0, t=30)
# )

# fig.show()

