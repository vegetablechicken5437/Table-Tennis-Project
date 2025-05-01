import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from calculation_3D import *
from itertools import combinations

def fit_and_plot_offset_plane(offsets, scale_factor=1.2, r=20):

    thres = 0.5

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        marker=dict(size=6, color='black'),
        name='Origin'
    ))

    # 過濾 None
    offsets_clean = offsets[~np.isnan(offsets).any(axis=1)]

    if len(offsets_clean) < 3:
        print("❗需要至少三個有效點才能擬合平面")
        return fig, np.array([np.nan, np.nan, np.nan]), offsets

    # 擬合平面
    centroid = offsets_clean.mean(axis=0)
    u, s, vh = np.linalg.svd(offsets_clean - centroid)
    normal = vh[-1]
    x_axis = vh[0]
    y_axis = vh[1]

    # 計算 filtered_offsets
    filtered_offsets = []
    for o in offsets:
        if o[0] == np.nan:
            filtered_offsets.append(np.array([np.nan, np.nan, np.nan]))
            continue
        vec = np.array(o) - centroid
        proj_x = np.dot(vec, x_axis)
        proj_y = np.dot(vec, y_axis)
        proj_len = np.sqrt(proj_x**2 + proj_y**2)
        if proj_len < r * thres:
            filtered_offsets.append(np.array([np.nan, np.nan, np.nan]))
        else:
            filtered_offsets.append(np.array(o))

    # 畫線條
    for i, o in enumerate(filtered_offsets):
        if np.any(np.isnan(o)):
            fig.add_trace(go.Scatter3d(
                x=[0, offsets[i][0]],
                y=[0, offsets[i][1]],
                z=[0, offsets[i][2]],
                mode='lines+markers',
                line=dict(color='rgba(200, 200, 200, 0.3)', width=3),
                marker=dict(size=3, color='rgba(200, 200, 200, 0.3)'),
                showlegend=False
            ))
        else:
            fig.add_trace(go.Scatter3d(
                x=[0, o[0]],
                y=[0, o[1]],
                z=[0, o[2]],
                mode='lines+markers',
                line=dict(color='red', width=3),
                marker=dict(size=3, color='red'),
                showlegend=False
            ))

    # 畫擬合平面
    proj_x = np.dot(offsets_clean - centroid, x_axis)
    proj_y = np.dot(offsets_clean - centroid, y_axis)
    max_x = np.max(np.abs(proj_x))
    max_y = np.max(np.abs(proj_y))
    plane_half_width = scale_factor * max(max_x, max_y)

    grid_range = np.linspace(-plane_half_width, plane_half_width, 2)
    grid_x, grid_y = np.meshgrid(grid_range, grid_range)
    grid_points = centroid + np.outer(grid_x.flatten(), x_axis) + np.outer(grid_y.flatten(), y_axis)
    px = grid_points[:, 0].reshape(2, 2)
    py = grid_points[:, 1].reshape(2, 2)
    pz = grid_points[:, 2].reshape(2, 2)

    fig.add_trace(go.Surface(
        x=px, y=py, z=pz,
        opacity=0.4,
        colorscale='Blues',
        showscale=False,
        name='Fitted Plane'
    ))

    # 畫法向量
    normal_arrow = normal / np.linalg.norm(normal) * plane_half_width * 0.8
    fig.add_trace(go.Scatter3d(
        x=[0, normal_arrow[0]],
        y=[0, normal_arrow[1]],
        z=[0, normal_arrow[2]],
        mode='lines+markers',
        line=dict(color='black', width=4),
        marker=dict(size=4, color='black'),
        name='Normal Vector'
    ))

    fig.update_layout(
        title='Offset Plane Fitting + Normal Vector',
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z')
    )

    return fig, normal, filtered_offsets



# def fit_and_plot_offset_plane(offsets, scale_factor=1.2):
#     """
#     擬合 offset 為平面，畫出該平面與原點出發的法向量。
#     平面大小自動依據 offset 分布決定。
#     """
#     # 建立初始圖
#     fig = go.Figure()
#     fig.add_trace(go.Scatter3d(
#         x=[0], y=[0], z=[0],
#         mode='markers',
#         marker=dict(size=6, color='black'),
#         name='Origin'
#     ))

#     # 畫出向量
#     for o in offsets:
#         if o is not None:
#             fig.add_trace(go.Scatter3d(
#                 x=[0, o[0]], 
#                 y=[0, o[1]], 
#                 z=[0, o[2]],
#                 mode='lines+markers',
#                 line=dict(color='red'),
#                 marker=dict(size=3, color='red'),
#                 showlegend=False
#             ))

#     offsets = np.array([pt for pt in offsets if pt is not None])
#     if len(offsets) < 3:
#         print("❗需要至少三個有效點才能擬合平面")
#         return fig, np.array([np.nan, np.nan, np.nan])

#     # Step 1: 擬合平面 via SVD
#     centroid = offsets.mean(axis=0)
#     u, s, vh = np.linalg.svd(offsets - centroid)
#     normal = vh[-1]  # 法向量
#     x_axis = vh[0]   # 平面上的第一主軸
#     y_axis = vh[1]   # 平面上的第二主軸

#     # Step 2: 投影到平面，決定大小
#     proj_x = np.dot(offsets - centroid, x_axis)
#     proj_y = np.dot(offsets - centroid, y_axis)
#     max_x = np.max(np.abs(proj_x))
#     max_y = np.max(np.abs(proj_y))
#     plane_half_width = scale_factor * max(max_x, max_y)

#     # Step 3: 建立平面點
#     grid_range = np.linspace(-plane_half_width, plane_half_width, 2)
#     grid_x, grid_y = np.meshgrid(grid_range, grid_range)
#     grid_points = centroid + np.outer(grid_x.flatten(), x_axis) + np.outer(grid_y.flatten(), y_axis)
#     px = grid_points[:, 0].reshape(2, 2)
#     py = grid_points[:, 1].reshape(2, 2)
#     pz = grid_points[:, 2].reshape(2, 2)

#     if fig is None:
#         fig = go.Figure()

#     # Step 4: 畫出平面
#     fig.add_trace(go.Surface(
#         x=px, y=py, z=pz,
#         opacity=0.4,
#         colorscale='Blues',
#         showscale=False,
#         name='Fitted Plane'
#     ))

#     # Step 5: 畫出法向量（原點出發）
#     normal_arrow = normal / np.linalg.norm(normal) * plane_half_width * 0.8
#     fig.add_trace(go.Scatter3d(
#         x=[0, normal_arrow[0]],
#         y=[0, normal_arrow[1]],
#         z=[0, normal_arrow[2]],
#         mode='lines+markers',
#         line=dict(color='black', width=4),
#         marker=dict(size=4, color='black'),
#         name='Normal Vector'
#     ))

#     fig.update_layout(
#         title='Offset Plane Fitting + Normal Vector',
#         scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z')
#     )
#     # fig.show()

#     return fig, normal

if __name__ == "__main__":

    traj_3D = np.loadtxt('OUTPUT/0408/20250408_193842/traj_3D.txt')
    marks_3D = np.loadtxt('OUTPUT/0408/20250408_193842/marks_3D.txt')
    marks_3D = np.loadtxt(r"C:\Users\jason\Desktop\TableTennisProject\OUTPUT\0415\20250415_193043\marks_3D_transformed.txt")

    print(marks_3D)
    print(marks_3D[~np.isnan(marks_3D).any(axis=1)])

