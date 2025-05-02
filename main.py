import os
import cv2
import numpy as np
import json
from image_processor import *
from label_processer import *
from yolo_runner import *
from pick_corners import CornerPicker
from calculation_3D import *
from traj_processor import *
# from spin_calculation import *
from spin_axis_calculation_new import *
from spin_rate_calculation_new import *
from visualize_functions import *

PROCESS_IMAGE = False
CREATE_VIDEO = False
TRAIN = {'Ball':False, 'Logo':False}
INFERENCE = {'Ball':False, 'Logo':False}
CROP_BBOX = False
EXTRACT_2D_POINTS = True
PICK_CORNERS = False
GEN_VERIFY_VIDEO = False
CALCULATE_3D = True
CALCULATE_SPIN_RATE = True

all_sample_folder_name = '0412'
sample_folder_name = '20250412_152611'

ori_img_folder_path = os.path.join('CameraControl/bin/x64/TableTennisData/', all_sample_folder_name, sample_folder_name)    # 原影像資料夾路徑
processed_img_folder_path = os.path.join('ProcessedImages', all_sample_folder_name, sample_folder_name)    # 處理後的影像資料夾路徑
os.makedirs(processed_img_folder_path, exist_ok=True) 

ball_yolo_params = {'img_size':640, 'batch':16, 'epochs':100}
mark_yolo_params = {'img_size':128, 'batch':16, 'epochs':100}

output_folder_path = os.path.join('OUTPUT', all_sample_folder_name)
output_sample_folder_path = os.path.join('OUTPUT', all_sample_folder_name, sample_folder_name)
os.makedirs(output_sample_folder_path, exist_ok=True)

camParamsPath = "CameraCalibration/STEREO_IMAGES/cvCalibration_result.txt"

# 空氣動力學參數: [重力加速度 (m/s^2), 桌球質量 (kg), 空氣密度 (kg/m^3), 球的迎風面積 (m^2), 球半徑 (m), 阻力係數, 馬格努斯力係數]
aero_params = {'g':9.8, 'm':0.0027, 'rho':1.2, 'A':0.001256, 'r':0.02, 'Cd':0.5, 'Cm':1.23}
FPS = 225
dt = 1 / FPS  # 每一幀的時間間隔 (秒)

if __name__ == '__main__':

    # ----------------------------------------------------------------
    # Step 1 & 2: 分割影像(原始影像為左右合併)
    # ----------------------------------------------------------------
    processed_folder_path = os.path.join(processed_img_folder_path, 'enhanced_LR')
    processed_L_folder_path = os.path.join(processed_img_folder_path, 'enhanced_L')
    processed_R_folder_path = os.path.join(processed_img_folder_path, 'enhanced_R')
    for folder_path in (processed_folder_path, processed_L_folder_path, processed_R_folder_path):
        os.makedirs(folder_path, exist_ok=True)

    if PROCESS_IMAGE:
        print('🚀 增強與分割所有影像 ...')
        for image_file_name in tqdm(os.listdir(ori_img_folder_path)):
            image_path = os.path.join(ori_img_folder_path, image_file_name)

            img = cv2.imread(image_path)
            enhanced = enhance_image(img, 2, 30)
            imgL, imgR = split_image(enhanced)

            # cv2.imwrite(os.path.join(processed_folder_path, f"{os.path.splitext(image_file_name)[0]}_EN.jpg"), enhanced)
            cv2.imwrite(os.path.join(processed_folder_path, f"{os.path.splitext(image_file_name)[0]}_L.jpg"), imgL)
            cv2.imwrite(os.path.join(processed_folder_path, f"{os.path.splitext(image_file_name)[0]}_R.jpg"), imgR)

            # cv2.imwrite(os.path.join(processed_L_folder_path, f"{os.path.splitext(image_file_name)[0]}_L.jpg"), imgL)
            # cv2.imwrite(os.path.join(processed_R_folder_path, f"{os.path.splitext(image_file_name)[0]}_R.jpg"), imgR)

    # if CREATE_VIDEO:
    #     for folder_path in (processed_L_folder_path, processed_R_folder_path):
    #         createVideo(folder_path, f'{folder_path.split('/')[-1]}.mp4', fps=20)

    # ----------------------------------------------------------------
    # Step 7: # 透過UI介面手動選取球桌四角，定義世界坐標系
    # ----------------------------------------------------------------
    if PICK_CORNERS:
        picker = CornerPicker([], output_folder_path)
        picker.pick_corners(processed_folder_path)
        left_corners, right_corners = picker.left_corners, picker.right_corners
    # ----------------------------------------------------------------

    # ----------------------------------------------------------------
    # Step 3: YOLO偵測桌球(可選擇是否訓練和預測)
    # ----------------------------------------------------------------
    ball_yolo_folder = 'BallDetection_YOLOv5/yolov5'
    ball_detection_yolov5(ball_yolo_folder, ball_yolo_params, processed_folder_path, 
                          all_sample_folder_name, sample_folder_name, 
                          TRAIN, INFERENCE)
    # ----------------------------------------------------------------

    # ----------------------------------------------------------------
    # Step 4: 裁切bounding box 並輸出裁切圖片
    # ----------------------------------------------------------------
    ball_bbox_label_path = f'{ball_yolo_folder}/runs/detect/{all_sample_folder_name}/exp_{sample_folder_name}/labels'    # 偵測結果資料夾(含多條軌跡的偵測結果)
    cropped_balls_folder = os.path.join('Cropped_Balls', all_sample_folder_name, sample_folder_name)
    os.makedirs(cropped_balls_folder, exist_ok=True)
    
    if CROP_BBOX:
        all_bbox_xyxy = crop_bbox(processed_folder_path, ball_bbox_label_path, cropped_balls_folder)
        # print(all_bbox_xyxy)
        with open(f"{output_sample_folder_path}/all_bbox_xyxy.json", "w") as fp:
            json.dump(all_bbox_xyxy, fp)  
        print(f"已儲存 {output_sample_folder_path}/all_bbox_xyxy.json")

    # ----------------------------------------------------------------
    # Step 5: YOLO偵測Logo(可選擇是否訓練和預測)
    # ----------------------------------------------------------------
    mark_yolo_folder = 'LogoDetection_YOLOv8'
    logo_detection_yolov8(mark_yolo_folder, mark_yolo_params, cropped_balls_folder, 
                          all_sample_folder_name, sample_folder_name, 
                          TRAIN, INFERENCE)

    # ----------------------------------------------------------------
    # Step 6: 輸出球和logo在影像上的座標(每個frame都有左、右影像的球座標)
    # ----------------------------------------------------------------
    mark_poly_label_path = f'{mark_yolo_folder}/runs/segment/predict/{all_sample_folder_name}/{sample_folder_name}/labels'
    if EXTRACT_2D_POINTS:
        with open(f"{output_sample_folder_path}/all_bbox_xyxy.json", "r") as fp:
            all_bbox_xyxy = json.load(fp)  
        all_2D_centers = extract_2D_points(mark_poly_label_path, all_bbox_xyxy)

    # ----------------------------------------------------------------
    if GEN_VERIFY_VIDEO:
        ball_bbox_img_path = os.path.join(ball_yolo_folder, f'runs/detect/{all_sample_folder_name}/exp_{sample_folder_name}')
        mark_poly_img_path = f'{mark_yolo_folder}/runs/segment/predict/{all_sample_folder_name}/{sample_folder_name}'
        generate_verify_video(all_2D_centers, ball_bbox_img_path, mark_poly_img_path, output_path= f'{output_sample_folder_path}/verify_video.mp4')

    # ----------------------------------------------------------------
    # Step 7: # 計算3D座標
    # ----------------------------------------------------------------
    """
    all_2D_centers = {
                        "image-0000_L.txt": {0: (ball_center_x, ball_center_y), 
                                             1: (mark_o_center_x, mark_o_center_y)}, 
                        "image-0001_R.txt": {0: (ball_center_x, ball_center_y), 
                                             2: (mark_x_center_x, mark_x_center_y)}, 
                        "image-0002_L.txt": {0: (ball_center_x, ball_center_y)}, 
                        "image-0002_R.txt": {0: (ball_center_x, ball_center_y)}, 
                        ...
                     }
    """
    
    if CALCULATE_3D:
        camParams = read_calibration_file(camParamsPath)
        lb, rb, lmo, rmo, lmx, rmx = extract_centers(all_2D_centers, total_frames=500)

        print('🚀 計算3D座標中...')
        left_corners = np.loadtxt(f'{output_folder_path}/left_corners.txt')
        right_corners = np.loadtxt(f'{output_folder_path}/right_corners.txt')
        
        corners_3D, _, _ = myDLT(camParams, left_corners, right_corners)
        traj_3D, traj_reproj_error_L, traj_reproj_error_R = myDLT(camParams, lb, rb)
        # marks_o_3D, mo_reproj_error_L, mo_reproj_error_R = myDLT(camParams, lmo, rmo)
        # marks_x_3D, mx_reproj_error_L, mx_reproj_error_R = myDLT(camParams, lmx, rmx)

        # # 輸出 reprojection error 圖表
        # plot_reprojection_error(
        #     traj_reproj_error_L, traj_reproj_error_R,
        #     mo_reproj_error_L, mo_reproj_error_R,
        #     mx_reproj_error_L, mx_reproj_error_R,
        #     path = f'{output_sample_folder_path}/reprojection_errors.jpg'
        # )
        
        # # 將 mark_x 座標轉為 mark_o 儲存為 marks_3D
        # marks_3D = marks_o_3D
        # for i in range(len(marks_o_3D)):
        #     P_mark, C_ball = marks_x_3D[i], traj_3D[i]
        #     if P_mark[0] != np.nan:
        #         mark_o_3D = mark_x_to_mark_o(P_mark, C_ball)
        #         marks_3D[i] = mark_o_3D

        marks_3D = get_marks_3D(camParams, traj_3D, lmo, rmo, lmx, rmx)    # 根據球心座標和球面方程式計算標記3D座標

        # 轉換為自訂的坐標系
        corners_3D_transformed, _ = transform_coord_system(corners_3D, corners_3D)
        traj_3D_transformed, _ = transform_coord_system(traj_3D, corners_3D)
        marks_3D_transformed = shift_marks_by_trajectory(traj_3D, traj_3D_transformed, marks_3D)
        # marks_3D_transformed, _ = transform_coord_system(marks_3D, corners_3D)

        np.savetxt(f'{output_folder_path}/corners_3D_transformed.txt', corners_3D_transformed)
        np.savetxt(f'{output_sample_folder_path}/traj_3D_transformed.txt', traj_3D_transformed)
        np.savetxt(f'{output_sample_folder_path}/marks_3D_transformed.txt', marks_3D_transformed)

        traj_list = [traj_3D_transformed]
        mark_list = [marks_3D_transformed]
        plot_multiple_3d_trajectories_with_plane(traj_list, mark_list, corners_3D_transformed, None, output_html=f'{output_sample_folder_path}/traj_ori.html')

        # 找出包含軌跡的 frame 和 start_idx, end_idx 從頭尾檢查非空值
        traj_3D_transformed, start_idx, end_idx = extract_valid_trajectory(traj_3D_transformed)
        marks_3D_transformed = marks_3D_transformed[start_idx:end_idx+1]

        # 移除軌跡異常點 平滑軌跡 標記點隨平滑後的軌跡平移
        cleaned_traj = remove_velocity_outliers(traj_3D_transformed)    # Step 1: 移除異常速度點

        # 偵測碰撞點 並根據碰撞點切分軌跡和標記
        temp_smoothed_traj = kalman_smooth_with_interp(cleaned_traj, smooth_strength=1.0, extend_points=10)     # 暫時平滑軌跡 有助找出碰傳idx
        collisions = detect_table_tennis_collisions_sequential(temp_smoothed_traj, corners_3D_transformed, z_tolerance=500)
        traj_3D_segs = split_trajectory_by_collisions(cleaned_traj, collisions)
        marks_3D_segs = split_trajectory_by_collisions(marks_3D_transformed, collisions)

        # 切分後每段軌跡分開平滑
        for i in range(len(traj_3D_segs)):
            smoothed_traj = kalman_smooth_with_interp(traj_3D_segs[i], smooth_strength=1.0, extend_points=10)
            marks_3D_segs[i] = shift_marks_by_trajectory(traj_3D_segs[i], smoothed_traj, marks_3D_segs[i])
            traj_3D_segs[i] = smoothed_traj
            np.savetxt(f'{output_sample_folder_path}/smoothed_traj{i+1}.txt', traj_3D_segs[i])

        # # 輸出每段軌跡和標記(以不同顏色區分)
        # plot_multiple_3d_trajectories_with_plane(traj_3D_segs, marks_3D_segs, corners_3D_transformed, None, output_html=f'{output_sample_folder_path}/traj_segs.html')

    # ----------------------------------------------------------------
    # Step 8: # 計算旋轉速度
    # ----------------------------------------------------------------
    # 用空氣動力學計算轉速
    px_list, py_list, pz_list, time_segments, t_list = process_parabolics(traj_3D_segs, dt)

    rps_list, spin_axis_list = [], []
    for i, t in enumerate(t_list):
        px, py, pz = px_list[i], py_list[i], pz_list[i]
        rps, spin_axes = compute_angular_velocity_rps(t, px, py, pz, aero_params)      # 帶入拋物線計算轉速
        rps_list.append(rps)
        spin_axis_list.append(np.mean(spin_axes, axis=0))

    plot_trajectories_with_spin_axes(px_list, py_list, pz_list, traj_3D_segs, spin_axis_list, dt, 
                                     path=f"{output_sample_folder_path}/polynomial_curves.html")
    
    plot_angular_velocity_curves(time_segments, rps_list, 
                                 path=f"{output_sample_folder_path}/rps_aero.jpg")

    if CALCULATE_SPIN_RATE:
        spin_axis_list = []  
        for i in range(len(traj_3D_segs)):
            offsets = marks_3D_segs[i] - traj_3D_segs[i]        # 計算標記相對球心的位置

            plane, filtered_offsets = fit_offset_plane(offsets)     # 擬和旋轉軸
            spin_axis = plane['normal']
            spin_axis_list.append(spin_axis)

            # 刪除和旋轉軸偏差過大的標記點
            for j in range(len(filtered_offsets)):
                if np.isnan(filtered_offsets[j][0]):
                    marks_3D_segs[i][j] = np.array([np.nan, np.nan, np.nan])

            plot_spin_axis(offsets, filtered_offsets, plane, path=f"{output_sample_folder_path}/spin_axis_seg{i+1}.html")

        # 畫軌跡、標記、旋轉軸(分段用不同顏色區分)
        plot_multiple_3d_trajectories_with_plane(traj_3D_segs, marks_3D_segs, corners_3D_transformed, 
                                                 spin_axis_list, output_html=f'{output_sample_folder_path}/traj_segs.html')

# ------------------------------------------------------------------------------------------------------------------------------

        for i in range(len(traj_3D_segs)):
            spin_axis = spin_axis_list[i]
            if np.isnan(spin_axis[0]):      # 如果沒有足夠的標記座標(至少三個)可以擬和平面 跳過後續轉速計算
                continue
            
            candidate_rps_lists = calc_candidate_spin_rates(traj_3D_segs[i], marks_3D_segs[i], spin_axis, fps=FPS)

            traj_3D_segs[i] /= 1000    # mm轉為公尺

            # 計算每一幀的速度 (Ground Truth)
            velocity_gt = np.diff(traj_3D_segs[i], axis=0) * FPS    # 速度計算
            acceleration_gt = np.diff(velocity_gt, axis=0) * FPS    # 加速度計算

            # 設定模擬步數
            num_steps = len(traj_3D_segs[i])

            print(f"\n== Traj_{i+1} ==")
            # 計算四種旋轉速度回推的軌跡
            candidate_trajectories = []
            for j, candidate_rps_list in enumerate(candidate_rps_lists):
                rps = find_best_rps(candidate_rps_list)
                print(f"candidate_{j+1}: {rps}")
                candidate_traj = compute_trajectory_aero(velocity_gt[0], traj_3D_segs[i][0], rps, dt, num_steps, spin_axis, aero_params)
                candidate_trajectories.append(candidate_traj)

            trajectory_cw, trajectory_cw_extra, trajectory_ccw, trajectory_ccw_extra = candidate_trajectories
            draw_candidate_trajectories(traj_3D_segs[i], trajectory_cw, trajectory_cw_extra, trajectory_ccw, trajectory_ccw_extra, 
                                        f"{output_sample_folder_path}/candidate_trajectories_{i+1}.html")

            # print("Rotation Axis (Plane Normal):", spin_axis)
            # print(rps_cw, rps_cw_extra, rps_ccw, rps_ccw_extra)
