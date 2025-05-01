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

# def plot_multiple_3d_trajectories_with_plane(traj_list_3D, corners_3D, output_html='multi_traj_plot.html'):
#     """
#     Parameters:
#         traj_list_3D: List of trajectories, each trajectory is a list of (x, y, z) tuples.
#                       e.g., [[(x1,y1,z1), (x2,y2,z2)], [(x3,y3,z3), (x4,y4,z4)], ...]
#         corners_3D: List of 4 (x, y, z) tuples forming a rectangle.
#         output_html: Output file path for the interactive Plotly HTML.
#     """
#     colors = [
#         (255, 0, 0),      # 紅色
#         (255, 127, 0),    # 橙色
#         (255, 255, 0),    # 黃色
#         (0, 255, 0),      # 綠色
#         (0, 0, 255),      # 藍色
#         (75, 0, 130),     # 靛色
#         (148, 0, 211),     # 紫色
#         (255, 0, 255),
#         (255, 127, 80),
#         (112, 66, 20),
#         (0, 128, 128),
#         (0, 255, 255)
#     ]

#     fig = go.Figure()

#     # 繪製每一條軌跡
#     for idx, traj in enumerate(traj_list_3D):
#         traj = traj[~np.isnan(traj).any(axis=1)]
#         # print(len(traj))
#         if len(traj) == 0:
#             continue
#         x, y, z = zip(*traj)
#         color = f'rgb({colors[idx][0]}, {colors[idx][1]}, {colors[idx][2]})'

#         fig.add_trace(go.Scatter3d(
#             x=x, y=y, z=z,
#             mode='markers',
#             # mode='lines+markers',
#             line=dict(color=color, width=4),
#             marker=dict(size=3),
#             name=f'Trajectory {idx+1}'
#         ))

#     # 畫矩形平面
#     x_corners, y_corners, z_corners = zip(*corners_3D)
#     x_plane = list(x_corners) + [x_corners[0]]
#     y_plane = list(y_corners) + [y_corners[0]]
#     z_plane = list(z_corners) + [z_corners[0]]

#     fig.add_trace(go.Mesh3d(
#         x=x_corners,
#         y=y_corners,
#         z=z_corners,
#         i=[0, 0],
#         j=[1, 2],
#         k=[2, 3],
#         color='lightgreen',
#         opacity=0.5,
#         name='Rectangle Plane'
#     ))

#     fig.add_trace(go.Scatter3d(
#         x=x_plane,
#         y=y_plane,
#         z=z_plane,
#         mode='lines',
#         line=dict(color='green', width=6),
#         name='Rectangle Edge'
#     ))

#     fig.update_layout(
#         title='Multiple 3D Trajectories with Reference Plane',
#         scene=dict(
#             xaxis_title='X',
#             yaxis_title='Y',
#             zaxis_title='Z',
#             aspectmode='data'
#             ),
#     )

#     pio.write_html(fig, file=output_html, auto_open=False)
#     print(f"✅ 已輸出至：{output_html}")


import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

def plot_multiple_3d_trajectories_with_plane(
    traj_list,
    mark_list,
    corners_3D,
    rotation_axis_list=None,
    output_html='multi_traj_plot.html',
    axis_length=200  # mm
):
    traj_colors = [
        (255, 0, 0), (255, 127, 0), (255, 255, 0), (0, 255, 0),
        (0, 0, 255), (75, 0, 130), (148, 0, 211), (255, 0, 255),
        (255, 127, 80), (112, 66, 20), (0, 128, 128), (0, 255, 255)
    ]

    mark_colors = [
        (100, 100, 100), (100, 50, 200), (200, 150, 0), (0, 200, 100),
        (150, 0, 150), (0, 150, 200), (50, 50, 255), (150, 100, 50),
        (200, 0, 100), (0, 50, 150), (100, 150, 0), (255, 100, 100)
    ]

    purple_shades = [
        'rgb(160, 32, 240)', 'rgb(128, 0, 128)', 'rgb(186, 85, 211)',
        'rgb(138, 43, 226)', 'rgb(147, 112, 219)', 'rgb(153, 50, 204)'
    ]

    fig = go.Figure()

    for idx, (traj, marks) in enumerate(zip(traj_list, mark_list)):
        traj_color = f'rgb{traj_colors[idx % len(traj_colors)]}'
        mark_color = f'rgb{mark_colors[idx % len(mark_colors)]}'

        # 畫軌跡點
        valid_traj = traj[~np.isnan(traj).any(axis=1)]
        if len(valid_traj) > 0:
            x, y, z = zip(*valid_traj)
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=dict(size=3, color=traj_color),
                name=f'Trajectory {idx+1}'
            ))

        # 畫標記點
        valid_mark = marks[~np.isnan(marks).any(axis=1)]
        if len(valid_mark) > 0:
            x, y, z = zip(*valid_mark)
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=dict(size=3, color=mark_color, symbol='circle'),
                name=f'Mark {idx+1}'
            ))

        # 畫連線
        lines_x, lines_y, lines_z = [], [], []
        for t, m in zip(traj, marks):
            if not np.isnan(t).any() and not np.isnan(m).any():
                lines_x.extend([t[0], m[0], None])
                lines_y.extend([t[1], m[1], None])
                lines_z.extend([t[2], m[2], None])

        if lines_x:
            fig.add_trace(go.Scatter3d(
                x=lines_x, y=lines_y, z=lines_z,
                mode='lines',
                line=dict(color='black', width=2),
                name=f'Link {idx+1}'
            ))

        # 畫對應旋轉軸（在該軌跡中間）
        if rotation_axis_list and idx < len(rotation_axis_list):
            axis_vec = np.array(rotation_axis_list[idx])
            if np.linalg.norm(axis_vec) > 1e-6 and len(valid_traj) > 0:
                unit_axis = axis_vec / np.linalg.norm(axis_vec)
                # midpoint = np.mean(valid_traj, axis=0)
                midpoint = valid_traj[len(valid_traj)//2]
                end_point = midpoint + unit_axis * axis_length
                color = purple_shades[idx % len(purple_shades)]

                fig.add_trace(go.Scatter3d(
                    x=[midpoint[0], end_point[0]],
                    y=[midpoint[1], end_point[1]],
                    z=[midpoint[2], end_point[2]],
                    mode='lines+markers',
                    line=dict(color=color, width=5, dash='dash'),
                    marker=dict(size=4, color=color),
                    name=f'Rotation Axis {idx+1}'
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
        title='3D Trajectories + Marks + Plane + Rotation Axes',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        )
    )

    pio.write_html(fig, file=output_html, auto_open=False)
    print(f"✅ 已輸出至：{output_html}")




# def plot_multiple_3d_trajectories_with_plane(traj_list, mark_list, corners_3D, output_html='multi_traj_plot.html'):
#     """
#     Parameters:
#         traj_list: List of numpy arrays (Nx3) representing trajectories.
#         mark_list: List of numpy arrays (Nx3) representing corresponding marks.
#         corners_3D: List of 4 (x, y, z) tuples forming a rectangle.
#         output_html: Output HTML path for Plotly visualization.
#     """
#     traj_colors = [
#         (255, 0, 0), (255, 127, 0), (255, 255, 0), (0, 255, 0),
#         (0, 0, 255), (75, 0, 130), (148, 0, 211), (255, 0, 255),
#         (255, 127, 80), (112, 66, 20), (0, 128, 128), (0, 255, 255)
#     ]

#     mark_colors = [
#         (100, 100, 100), (100, 50, 200), (200, 150, 0), (0, 200, 100),
#         (150, 0, 150), (0, 150, 200), (50, 50, 255), (150, 100, 50),
#         (200, 0, 100), (0, 50, 150), (100, 150, 0), (255, 100, 100)
#     ]

#     fig = go.Figure()

#     for idx, (traj, marks) in enumerate(zip(traj_list, mark_list)):
#         traj_color = f'rgb{traj_colors[idx % len(traj_colors)]}'
#         mark_color = f'rgb{mark_colors[idx % len(mark_colors)]}'

#         # 畫軌跡點
#         valid_traj = traj[~np.isnan(traj).any(axis=1)]
#         if len(valid_traj) > 0:
#             x, y, z = zip(*valid_traj)
#             fig.add_trace(go.Scatter3d(
#                 x=x, y=y, z=z,
#                 mode='markers',
#                 marker=dict(size=3, color=traj_color),
#                 name=f'Trajectory {idx+1}'
#             ))

#         # 畫標記點
#         valid_mark = marks[~np.isnan(marks).any(axis=1)]
#         if len(valid_mark) > 0:
#             x, y, z = zip(*valid_mark)
#             fig.add_trace(go.Scatter3d(
#                 x=x, y=y, z=z,
#                 mode='markers',
#                 marker=dict(size=3, color=mark_color, symbol='circle'),
#                 name=f'Mark {idx+1}'
#             ))

#         # 畫連線（黑色）
#         lines_x, lines_y, lines_z = [], [], []
#         for t, m in zip(traj, marks):
#             if not np.isnan(t).any() and not np.isnan(m).any():
#                 lines_x.extend([t[0], m[0], None])
#                 lines_y.extend([t[1], m[1], None])
#                 lines_z.extend([t[2], m[2], None])

#         if lines_x:
#             fig.add_trace(go.Scatter3d(
#                 x=lines_x, y=lines_y, z=lines_z,
#                 mode='lines',
#                 line=dict(color='black', width=2),
#                 name=f'Link {idx+1}'
#             ))

#     # 畫矩形平面
#     x_corners, y_corners, z_corners = zip(*corners_3D)
#     x_plane = list(x_corners) + [x_corners[0]]
#     y_plane = list(y_corners) + [y_corners[0]]
#     z_plane = list(z_corners) + [z_corners[0]]

#     fig.add_trace(go.Mesh3d(
#         x=x_corners,
#         y=y_corners,
#         z=z_corners,
#         i=[0, 0],
#         j=[1, 2],
#         k=[2, 3],
#         color='lightgreen',
#         opacity=0.5,
#         name='Rectangle Plane'
#     ))

#     fig.add_trace(go.Scatter3d(
#         x=x_plane,
#         y=y_plane,
#         z=z_plane,
#         mode='lines',
#         line=dict(color='green', width=6),
#         name='Rectangle Edge'
#     ))

#     fig.update_layout(
#         title='3D Trajectories + Marks + Reference Plane',
#         scene=dict(
#             xaxis_title='X',
#             yaxis_title='Y',
#             zaxis_title='Z',
#             aspectmode='data'
#         )
#     )

#     pio.write_html(fig, file=output_html, auto_open=False)
#     print(f"✅ 已輸出至：{output_html}")


def plot_reprojection_error(
    traj_reproj_error_L, traj_reproj_error_R,
    mo_reproj_error_L, mo_reproj_error_R,
    mx_reproj_error_L, mx_reproj_error_R
):
    # 將所有誤差 list 與標籤打包
    error_lists = [
        (traj_reproj_error_L, "Traj Left"),
        (mo_reproj_error_L, "Mark O Left"),
        (mx_reproj_error_L, "Mark X Left"),
        (traj_reproj_error_R, "Traj Right"),
        (mo_reproj_error_R, "Mark O Right"),
        (mx_reproj_error_R, "Mark X Right")
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, (errors, title) in enumerate(error_lists):
        # 將誤差 list 轉為 np array 並排除空值（如 None 或 nan）
        clean_errors = np.array([e for e in errors if e is not None and not np.isnan(e)])

        ax = axes[idx]
        ax.hist(clean_errors, bins=30, color='skyblue', edgecolor='black')
        avg_error = np.mean(clean_errors) if len(clean_errors) > 0 else 0
        ax.set_title(f"{title}\nAvg Error = {avg_error:.3f} px")
        ax.set_xlabel("Reprojection Error (px)")
        ax.set_ylabel("Frequency")

    plt.tight_layout()
    # plt.show()
    return fig

def plot_angular_velocity_curves(t_list, rps_list, path):

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    plt.figure(figsize=(12, 6))
    for i, (t, rps, color) in enumerate(zip(t_list, rps_list, colors), 1):
        avg_rps = np.mean(rps)
        plt.plot(t, rps, label=f'Traj {i} (avg = {avg_rps:.2f} rps)', color=color)

    plt.xlabel("Time (s)")
    plt.ylabel("Angular Velocity (rps)")
    plt.title("Angular Velocity (rps) from Polynomial-Fitted Trajectories")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)

# Function：用 plotly 畫出 3D 軌跡與旋轉軸（輸入為已擬合好的 px, py, pz）
def plot_trajectories_with_spin_axes_plotly(px_list, py_list, pz_list, original_trajs, aero_params, dt, path=None):

    plotly_colors = ['blue', 'orange', 'green', 'red']

    g, m, rho, A, r, Cd, Cm = aero_params.values()
    fig = go.Figure()

    for i, (px, py, pz, traj) in enumerate(zip(px_list, py_list, pz_list, original_trajs)):
        t = np.arange(len(traj)) * dt
        t_fine = np.linspace(t[0], t[-1], 300)
        x_fit = px(t_fine)
        y_fit = py(t_fine)
        z_fit = pz(t_fine)

        # 擬合曲線
        fig.add_trace(go.Scatter3d(
            x=x_fit, y=y_fit, z=z_fit,
            mode='lines',
            line=dict(color=plotly_colors[i]),
            name=f'Traj {i+1} (fit)'
        ))

        # 原始資料點
        traj_m = traj / 1000.0
        fig.add_trace(go.Scatter3d(
            x=traj_m[:, 0], y=traj_m[:, 1], z=traj_m[:, 2],
            mode='markers',
            marker=dict(size=2, color=plotly_colors[i]),
            name=f'Traj {i+1} (raw)',
            opacity=0.3
        ))

        # 旋轉軸方向（使用中點）
        t0 = t[len(t) // 2]
        v0 = np.array([px.deriv(1)(t0), py.deriv(1)(t0), pz.deriv(1)(t0)])
        a0 = np.array([px.deriv(2)(t0), py.deriv(2)(t0), pz.deriv(2)(t0)])
        vnorm = np.linalg.norm(v0)
        Fd = -0.5 * Cd * rho * A * vnorm * v0
        Fnet = m * a0 - m * g - Fd
        omega_vec = np.cross(Fnet, v0) / (vnorm ** 2)
        omega_vec *= 3 / (4 * Cm * np.pi * r**3 * rho)
        omega_vec = omega_vec / np.linalg.norm(omega_vec) * 0.3  # normalize and scale

        origin = np.array([px(t0), py(t0), pz(t0)])
        arrow_end = origin + omega_vec

        fig.add_trace(go.Scatter3d(
            x=[origin[0], arrow_end[0]],
            y=[origin[1], arrow_end[1]],
            z=[origin[2], arrow_end[2]],
            mode='lines+markers',
            line=dict(color=plotly_colors[i], width=6),
            marker=dict(size=4, color=plotly_colors[i]),
            name=f'Traj {i+1} Spin Axis'
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)'
        ),
        title="Fitted 3D Trajectories with Estimated Spin Axes (Plotly)",
        template="plotly_white"
    )

    pio.write_html(fig, file=path, auto_open=False)

