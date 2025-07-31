import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.io as pio
import random

def plot_candidate_trajectories(traj_3D, candidate_trajectories, spin_axis, corners_3D, save_path=None):
    """
    使用 Plotly 繪製桌球運動軌跡，並儲存圖像為 HTML 檔案。
    """
    purple_shades = [
        'rgb(160, 32, 240)', 'rgb(128, 0, 128)', 'rgb(186, 85, 211)',
        'rgb(138, 43, 226)', 'rgb(147, 112, 219)', 'rgb(153, 50, 204)'
    ]
    axis_length = 0.2

    trajectory_cw, trajectory_cw_extra, trajectory_ccw, trajectory_ccw_extra = candidate_trajectories

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

    # 畫對應旋轉軸（在該軌跡中間）
    axis_vec = np.array(spin_axis)
    if np.linalg.norm(axis_vec) > 1e-6 and len(traj_3D) > 0:
        unit_axis = axis_vec / np.linalg.norm(axis_vec)
        midpoint = traj_3D[len(traj_3D)//2]
        end_point = midpoint + unit_axis * axis_length
        color = purple_shades[0]

        fig.add_trace(go.Scatter3d(
            x=[midpoint[0], end_point[0]],
            y=[midpoint[1], end_point[1]],
            z=[midpoint[2], end_point[2]],
            mode='lines+markers',
            line=dict(color=color, width=5, dash='dash'),
            marker=dict(size=4, color=color),
            name=f'Rotation Axis'
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

    # 設定圖表標籤與視角
    fig.update_layout(
        title="Aerodynamics-Adjusted Table Tennis Ball Trajectories",
        scene=dict(
            xaxis_title="X Position (m)",
            yaxis_title="Y Position (m)",
            zaxis_title="Z Position (m)",
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # 儲存為 HTML 檔案
    fig.write_html(save_path)

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
                mode='lines+markers',
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

def plot_reprojection_error(traj_reproj_error_L, traj_reproj_error_R, m_reproj_error_L, m_reproj_error_R, path=None):
    # # 將所有誤差 list 與標籤打包
    # error_lists = [
    #     (traj_reproj_error_L, "Traj Left"),
    #     (traj_reproj_error_R, "Traj Right"),
    #     (m_reproj_error_L, "Mark Left"),
    #     (m_reproj_error_R, "Mark Right")
    # ]

    # traj_combined = np.concatenate(traj_reproj_error_L, traj_reproj_error_R)
    # mark_combined = np.concatenate(m_reproj_error_L, m_reproj_error_R)

    traj_combined = np.array(traj_reproj_error_L + traj_reproj_error_R)
    mark_combined = np.array(m_reproj_error_L + m_reproj_error_R)

    error_lists = [
        (traj_combined, "Trajectory"),
        (mark_combined, "Marker")
    ]

    # 移除負值
    traj_combined = traj_combined[traj_combined > 0]
    mark_combined = mark_combined[mark_combined > 0]

    # 計算合併後的平均
    traj_mean = traj_combined.mean()
    mark_mean = mark_combined.mean()

    # fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # axes = axes.flatten()

    # bins = np.linspace(0, 5, 30)

    # for idx, (errors, title) in enumerate(error_lists):
    #     # 將誤差 list 轉為 np array 並排除空值（如 None 或 nan）
    #     clean_errors = np.array([e for e in errors if e is not None and not np.isnan(e)])

    #     ax = axes[idx]
    #     ax.hist(clean_errors, bins=bins, color='skyblue', edgecolor='black')
    #     # avg_error = np.mean(clean_errors) if len(clean_errors) > 0 else 0
    #     # ax.set_title(f"{title}\nAvg Error = {avg_error:.3f} px")
    #     ax.set_title(title)
    #     ax.axvline(clean_errors.mean(), color='red', linestyle='--', label=f'Avg Error = {clean_errors.mean():.3f} px')
    #     ax.set_xlabel("Reprojection Error (px)")
    #     ax.set_ylabel("Frequency")
    #     ax.legend()

    # plt.tight_layout()
    # # plt.show()
    # if path:
    #     fig.savefig(path)
    #     print(f"✅ 已輸出至：{path}")

    # 統一 bin 和座標軸範圍
    bins = np.linspace(0, 4.5, 30)
    y_max = max(
        np.histogram(traj_combined, bins=bins)[0].max(),
        np.histogram(mark_combined, bins=bins)[0].max()
    )

    # 畫圖：Trajectory
    plt.figure(figsize=(8, 5))
    plt.hist(traj_combined, bins=bins, color='skyblue', edgecolor='black')
    plt.axvline(traj_mean, color='red', linestyle='--', label=f'Avg Error = {traj_mean:.3f} px')
    plt.title("Trajectory", fontsize=18)
    plt.xlabel("Reprojection Error (px)", fontsize=15)
    plt.ylabel("Frequency", fontsize=15)
    plt.xlim(0, 4.5)
    plt.ylim(0, y_max + 5)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12) 
    plt.yticks(fontsize=12)
    plt.tight_layout()
    if path:
        plt.savefig(path + '/reproj_1.jpg')
        print(f"✅ 已輸出至：{path + '/reproj_1.jpg'}")

    # 畫圖：Marker
    plt.figure(figsize=(8, 5))
    plt.hist(mark_combined, bins=bins, color='lightgreen', edgecolor='black')
    plt.axvline(mark_mean, color='red', linestyle='--', label=f'Avg Error = {mark_mean:.3f} px')
    plt.title("Markers", fontsize=18)
    plt.xlabel("Reprojection Error (px)", fontsize=15)
    plt.ylabel("Frequency", fontsize=15)
    plt.xlim(0, 4.5)
    plt.ylim(0, y_max + 5)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12) 
    plt.yticks(fontsize=12)
    plt.tight_layout()
    if path:
        plt.savefig(path + '/reproj_2.jpg')
        print(f"✅ 已輸出至：{path + '/reproj_2.jpg'}")

    

def plot_spin_axis_with_fit_plane(offsets, filtered_offsets, plane, path=None, scale_factor=1.2):

    normal, x_axis, y_axis = plane['normal'], plane['x_axis'], plane['y_axis']

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        marker=dict(size=6, color='black'),
        name='Origin'
    ))

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

    offsets_clean = offsets[~np.isnan(offsets).any(axis=1)]
    centroid = offsets_clean.mean(axis=0)

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

    # 輸出旋轉軸圖
    if path:
        pio.write_html(fig, file=path, auto_open=False)
        print(f"✅ 已輸出至：{path}")

def plot_projected_marks_on_plane_all_frame(valid_data, plane_normal, save_html=None):
    # 建立 Frames
    frames = []
    for i, v1, v2, theta in valid_data:
        rotation_axis_end = plane_normal / np.linalg.norm(plane_normal) * 0.8

        frames.append(go.Frame(
            data=[
                go.Scatter3d(x=[0, v1[0]], y=[0, v1[1]], z=[0, v1[2]],
                             mode='lines+markers', line=dict(color='blue', width=8), name='v1'),
                go.Scatter3d(x=[0, v2[0]], y=[0, v2[1]], z=[0, v2[2]],
                             mode='lines+markers', line=dict(color='red', width=8), name='v2'),
                go.Scatter3d(x=[0, rotation_axis_end[0]], y=[0, rotation_axis_end[1]], z=[0, rotation_axis_end[2]],
                             mode='lines+markers', line=dict(color='green', width=6, dash='dash'), name='rotation_axis'),
            ],
            name=str(i),
            layout=go.Layout(
                annotations=[
                    dict(
                        showarrow=False,
                        text=f"<b>Frame: {i} | θ: {theta:.2f}°</b>",
                        xref="paper", yref="paper",
                        x=0.01, y=0.95,
                        font=dict(size=20, color="black"),
                        align="left",
                        bordercolor="black",
                        borderwidth=0
                    )
                ]
            )
        ))

    # 初始資料與 Slider
    fig = go.Figure(
        data=frames[0].data,
        layout=go.Layout(
            title=dict(text="<b>Spin Vector Visualization</b>", font=dict(size=24)),
            scene=dict(
                xaxis=dict(range=[-1, 1], title='X'),
                yaxis=dict(range=[-1, 1], title='Y'),
                zaxis=dict(range=[-1, 1], title='Z'),
                aspectmode='cube'
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                y=1.15,
                x=0,
                xanchor="left",
                buttons=[
                    dict(label="▶ Play", method="animate",
                         args=[None, {"frame": {"duration": 700, "redraw": True}, "fromcurrent": True}]),
                    dict(label="⏸ Pause", method="animate",
                         args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}])
                ]
            )],
            sliders=[{
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "prefix": "Frame: ",
                    "font": {"size": 18}
                },
                "pad": {"t": 30},
                "steps": [{
                    "method": "animate",
                    "label": str(i),
                    "args": [[str(i)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}]
                } for i, *_ in valid_data]
            }]
        ),
        frames=frames
    )

    fig.write_html(save_html)
    print(f"✅ 動畫已儲存至 {save_html}")


def plot_all_3d_trajectories(traj_list, corners_3D, output_html='multi_traj_plot.html'):
    traj_colors = [
        (255, 0, 0), (255, 127, 0), (255, 255, 0), (0, 255, 0),
        (0, 0, 255), (75, 0, 130), (148, 0, 211), (255, 0, 255),
        (255, 127, 80), (112, 66, 20), (0, 128, 128), (0, 255, 255)
    ]

    fig = go.Figure()

    for idx, traj in enumerate(traj_list):
        traj_color = f'rgb{traj_colors[idx % len(traj_colors)]}'

        # 畫軌跡點
        valid_traj = traj[~np.isnan(traj).any(axis=1)]
        if len(valid_traj) > 0:
            x, y, z = zip(*valid_traj)
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines+markers',
                marker=dict(size=3, color=traj_color),
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
        title='3D Trajectories',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        )
    )

    pio.write_html(fig, file=output_html, auto_open=False)
    print(f"✅ 已輸出至：{output_html}")


