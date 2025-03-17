import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

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

# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# def plot_trajectory_with_spin(traj_3D, plane_normal, best_rps, save_path):
#     """
#     使用 Matplotlib 繪製軌跡，並在每個點上標示旋轉速度 (RPS)。
    
#     :param traj_3D: 桌球的 3D 軌跡 (numpy array)
#     :param plane_normal: 旋轉平面的法向量 (numpy array)
#     :param best_rps: 最佳匹配的旋轉速度 (float)
#     :param save_path: 儲存圖片的路徑
#     """
#     fig = plt.figure(figsize=(8, 8))
#     ax = fig.add_subplot(111, projection='3d')
    
#     # 繪製桌球軌跡
#     ax.plot(traj_3D[:, 0], traj_3D[:, 1], traj_3D[:, 2], 'k-', linewidth=3, label="Ball Trajectory")
    
#     # 在每個點上標示旋轉速度方向
#     for i in range(len(traj_3D)):
#         pos = traj_3D[i]
#         spin_vector = best_rps * plane_normal  # 旋轉速度向量
        
#         ax.quiver(pos[0], pos[1], pos[2], 
#                   spin_vector[0], spin_vector[1], spin_vector[2], 
#                   color='blue', length=0.1, normalize=True)
    
#     # 設定圖表標籤
#     ax.set_xlabel("X Position (m)")
#     ax.set_ylabel("Y Position (m)")
#     ax.set_zlabel("Z Position (m)")
#     ax.set_title("Table Tennis Ball Trajectory with Spin Vectors")
#     ax.legend()
    
#     # 儲存圖片
#     # plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.show()
