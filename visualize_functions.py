import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.io as pio
import random

def draw_spin_axis(translated_logos, plane_normal):
    # 生成擬合平面
    grid_x, grid_y = np.meshgrid(np.linspace(-0.1, 0.1, 10), np.linspace(-0.1, 0.1, 10))
    grid_z = (-plane_normal[0] * grid_x - plane_normal[1] * grid_y) / plane_normal[2]

    # 繪製3D圖
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(translated_logos[:, 0], translated_logos[:, 1], translated_logos[:, 2], c='r', label="Translated Logos")
    ax.plot_surface(grid_x, grid_y, grid_z, color='cyan', alpha=0.5, edgecolor='k')
    ax.quiver(0, 0, 0, plane_normal[0], plane_normal[1], plane_normal[2], color='g', length=0.2, linewidth=2, label="Corrected Plane Normal")
    ax.scatter(0, 0, 0, c='b', s=100, label="Origin (Ball Center)")

    # 標籤與視角設定
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Fitted Plane & Corrected Rotation Axis")
    ax.legend()
    ax.view_init(elev=20, azim=30)

    plt.show()

def draw_trajectories(traj_3D, trajectory_cw, trajectory_cw_extra, trajectory_ccw, trajectory_ccw_extra, save_path):
    """
    使用 Plotly 繪製桌球運動軌跡，並儲存圖像為 HTML 檔案。
    """
    fig = go.Figure()

    # 添加 Ground Truth
    fig.add_trace(go.Scatter3d(
        x=traj_3D[:, 0], y=traj_3D[:, 1], z=traj_3D[:, 2],
        mode='lines', line=dict(color='black', width=3),
        name="Ground Truth"
    ))

    # 添加計算軌跡
    fig.add_trace(go.Scatter3d(
        x=trajectory_cw[:, 0], y=trajectory_cw[:, 1], z=trajectory_cw[:, 2],
        mode='lines', line=dict(color='red', width=2),
        name="CW"
    ))
    fig.add_trace(go.Scatter3d(
        x=trajectory_cw_extra[:, 0], y=trajectory_cw_extra[:, 1], z=trajectory_cw_extra[:, 2],
        mode='lines', line=dict(color='red', dash='dash', width=2),
        name="CW + 360°"
    ))
    fig.add_trace(go.Scatter3d(
        x=trajectory_ccw[:, 0], y=trajectory_ccw[:, 1], z=trajectory_ccw[:, 2],
        mode='lines', line=dict(color='blue', width=2),
        name="CCW"
    ))
    fig.add_trace(go.Scatter3d(
        x=trajectory_ccw_extra[:, 0], y=trajectory_ccw_extra[:, 1], z=trajectory_ccw_extra[:, 2],
        mode='lines', line=dict(color='blue', dash='dash', width=2),
        name="CCW + 360°"
    ))

    # 設定圖表標籤與視角
    fig.update_layout(
        title="Aerodynamics-Adjusted Table Tennis Ball Trajectories",
        scene=dict(
            xaxis_title="X Position (m)",
            yaxis_title="Y Position (m)",
            zaxis_title="Z Position (m)",
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # 儲存為 HTML 檔案
    fig.write_html(save_path)

def plot_trajectory_with_spin(traj_3D, plane_normal, best_rps, save_path):
    """
    使用 Plotly 繪製軌跡，並在每個點上標示旋轉速度 (RPS)。
    
    :param traj_3D: 桌球的 3D 軌跡 (numpy array)
    :param plane_normal: 旋轉平面的法向量 (numpy array)
    :param best_rps: 最佳匹配的旋轉速度 (float)
    :param save_path: 儲存 HTML 圖檔的路徑
    """
    fig = go.Figure()

    # 添加桌球軌跡
    fig.add_trace(go.Scatter3d(
        x=traj_3D[:, 0], y=traj_3D[:, 1], z=traj_3D[:, 2],
        mode='lines', line=dict(color='black', width=3),
        name="Ball Trajectory"
    ))

    # # 在每個點上標示旋轉速度方向
    # for i in range(len(traj_3D)):
    #     pos = traj_3D[i]
    #     spin_vector = best_rps * plane_normal  # 旋轉速度向量

    #     fig.add_trace(go.Cone(
    #         x=[pos[0]], y=[pos[1]], z=[pos[2]],
    #         u=[spin_vector[0]], v=[spin_vector[1]], w=[spin_vector[2]],
    #         sizemode="absolute", sizeref=0.1,
    #         anchor="tip", colorscale="Blues", name=f"Spin {i}"
    #     ))

    # 在每個點上標示旋轉速度方向（用線條表示）
    for i in range(len(traj_3D)):
        pos = traj_3D[i]
        spin_vector = best_rps * plane_normal * 0.01  # 調整長度，縮短旋轉向量
        
        fig.add_trace(go.Scatter3d(
            x=[pos[0], pos[0] + spin_vector[0]],
            y=[pos[1], pos[1] + spin_vector[1]],
            z=[pos[2], pos[2] + spin_vector[2]],
            mode='lines+text',
            line=dict(color='blue', width=2),
            name=f"Spin {i}",
            showlegend=False,  # 避免圖例過多
        ))

    # 設定圖表標籤與視角
    fig.update_layout(
        title="Table Tennis Ball Trajectory with Spin Vectors",
        scene=dict(
            xaxis_title="X Position (m)",
            yaxis_title="Y Position (m)",
            zaxis_title="Z Position (m)",
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # 儲存為 HTML 檔案
    fig.write_html(save_path)

def plot_multiple_3d_trajectories_with_plane(traj_list_3D, corners_3D, output_html='multi_traj_plot.html'):
    """
    Parameters:
        traj_list_3D: List of trajectories, each trajectory is a list of (x, y, z) tuples.
                      e.g., [[(x1,y1,z1), (x2,y2,z2)], [(x3,y3,z3), (x4,y4,z4)], ...]
        corners_3D: List of 4 (x, y, z) tuples forming a rectangle.
        output_html: Output file path for the interactive Plotly HTML.
    """
    colors = [
        (255, 0, 0),      # 紅色
        (255, 127, 0),    # 橙色
        (255, 255, 0),    # 黃色
        (0, 255, 0),      # 綠色
        (0, 0, 255),      # 藍色
        (75, 0, 130),     # 靛色
        (148, 0, 211)     # 紫色
    ]


    fig = go.Figure()

    # 繪製每一條軌跡
    for idx, traj in enumerate(traj_list_3D):
        x, y, z = zip(*traj)
        color = f'rgb({colors[idx][0]}, {colors[idx][1]}, {colors[idx][2]})'

        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            # mode='lines+markers',
            line=dict(color=color, width=4),
            marker=dict(size=3),
            name=f'Trajectory {idx+1}'
        ))

    # 畫矩形平面
    x_corners, y_corners, z_corners = zip(*corners_3D)
    x_plane = list(x_corners) + [x_corners[0]]
    y_plane = list(y_corners) + [y_corners[0]]
    z_plane = list(z_corners) + [z_corners[0]]

    fig.add_trace(go.Mesh3d(
        x=x_corners,
        y=y_corners,
        z=z_corners,
        i=[0, 0],
        j=[1, 2],
        k=[2, 3],
        color='lightgreen',
        opacity=0.5,
        name='Rectangle Plane'
    ))

    fig.add_trace(go.Scatter3d(
        x=x_plane,
        y=y_plane,
        z=z_plane,
        mode='lines',
        line=dict(color='green', width=6),
        name='Rectangle Edge'
    ))

    fig.update_layout(
        title='Multiple 3D Trajectories with Reference Plane',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            xaxis=dict(
                range=[-3, 3],
                tickmode='linear',
                dtick=1
            ),
            yaxis=dict(
                range=[-3, 3],
                tickmode='linear',
                dtick=1
            ),
            zaxis=dict(
                range=[-3, 3],
                tickmode='linear',
                dtick=1
            ),
            # ⬇️ 關鍵：手動設定每軸空間比例
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=1)
            ),
            margin=dict(l=0, r=0, b=0, t=40)
    )

    pio.write_html(fig, file=output_html, auto_open=False)
    print(f"✅ 已輸出至：{output_html}")
