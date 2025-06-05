import os
import cv2
import numpy as np
from image_processor import *
from label_processer import *
from yolo_runner import *
from pick_corners import CornerPicker
from calculation_3D import *
from traj_processor import *
from spin_axis_calculation_new import *
from spin_rate_calculation_new import *
from visualize_functions import *

PROCESS_IMAGE = True

TRAIN_BALL_DETECT_MODEL = False
TRAIN_MARK_DETECT_MODEL = False
INFERENCE_BALL = True
INFERENCE_MARK = True

CROP_BBOX = True
PICK_CORNERS = True
GEN_VERIFY_VIDEO = True

CALCULATE_3D = True
CALCULATE_SPIN_RATE = True

all_sample_folder_name = '0527'
sample_folder_name = '20250527_215347'

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
    # 分割影像(原始影像為左右合併)
    # ----------------------------------------------------------------
    processed_folder_path = os.path.join(processed_img_folder_path, 'enhanced_LR')
    os.makedirs(processed_folder_path, exist_ok=True)

    if PROCESS_IMAGE:
        print('🚀 增強與分割所有影像 ...')
        for image_file_name in tqdm(os.listdir(ori_img_folder_path)):
            image_path = os.path.join(ori_img_folder_path, image_file_name)

            img = cv2.imread(image_path)
            enhanced = enhance_image(img, 2, 30)
            imgL, imgR = split_image(enhanced)

            cv2.imwrite(os.path.join(processed_folder_path, f"{os.path.splitext(image_file_name)[0]}_L.jpg"), imgL)
            cv2.imwrite(os.path.join(processed_folder_path, f"{os.path.splitext(image_file_name)[0]}_R.jpg"), imgR)

    # ----------------------------------------------------------------
    # 透過UI介面手動選取球桌四角，定義世界坐標系
    # ----------------------------------------------------------------
    if PICK_CORNERS and not os.path.exists(f"{output_folder_path}/corners_3D_transformed.txt"):
        picker = CornerPicker([], output_folder_path)
        picker.pick_corners(processed_folder_path)
        left_corners, right_corners = picker.left_corners, picker.right_corners
    # ----------------------------------------------------------------

    # ----------------------------------------------------------------
    # YOLO偵測桌球(可選擇是否訓練和預測)
    # ----------------------------------------------------------------
    ball_yolo_folder = 'BallDetection_YOLOv11'
    if TRAIN_BALL_DETECT_MODEL:
        ball_detection_yolov11_training(ball_yolo_folder, ball_yolo_params)
    if INFERENCE_BALL:
        ball_detection_yolov11_inferencing(
                                            ball_yolo_folder=ball_yolo_folder, 
                                            ball_yolo_params=ball_yolo_params, 
                                            input_folder=processed_folder_path, 
                                            all_sample_folder_name=all_sample_folder_name, 
                                            sample_folder_name=sample_folder_name
                                           )

    # ----------------------------------------------------------------
    # 裁切bounding box 並輸出裁切圖片
    # ----------------------------------------------------------------
    ball_bbox_label_path = f'{ball_yolo_folder}/runs/detect/predict/{all_sample_folder_name}/{sample_folder_name}/labels'    # 偵測結果資料夾(含多條軌跡的偵測結果)
    cropped_balls_folder = os.path.join('Cropped_Balls', all_sample_folder_name, sample_folder_name)
    bbox_xyxy_path=f"{output_sample_folder_path}/all_bbox_xyxy.json"
    os.makedirs(cropped_balls_folder, exist_ok=True)
    
    if CROP_BBOX:
        crop_bbox(
                    img_folder=processed_folder_path, 
                    ball_bbox_label_path=ball_bbox_label_path, 
                    output_folder=cropped_balls_folder,
                    bbox_xyxy_path=bbox_xyxy_path
                  )

    # ----------------------------------------------------------------
    # YOLO偵測Logo(可選擇是否訓練和預測)
    # ----------------------------------------------------------------
    mark_yolo_folder = 'MarkDetection_YOLOv11'
    if TRAIN_MARK_DETECT_MODEL:
        mark_detection_yolov11_training(mark_yolo_folder, mark_yolo_params)
    if INFERENCE_MARK:
        mark_detection_yolov11_inferencing(
                                            mark_yolo_folder=mark_yolo_folder, 
                                            mark_yolo_params=mark_yolo_params, 
                                            input_folder=cropped_balls_folder, 
                                            all_sample_folder_name=all_sample_folder_name, 
                                            sample_folder_name=sample_folder_name
                                           )
    # ----------------------------------------------------------------
    # 輸出球和logo在影像上的座標(每個frame都有左、右影像的球座標)
    # ----------------------------------------------------------------
    mark_poly_label_path = f'{mark_yolo_folder}/runs/segment/predict/{all_sample_folder_name}/{sample_folder_name}/labels'
    all_2D_centers = extract_2D_points(mark_poly_label_path, bbox_xyxy_path)
    # ----------------------------------------------------------------

    if GEN_VERIFY_VIDEO:
        ball_bbox_img_path = f'{ball_yolo_folder}/runs/detect/predict/{all_sample_folder_name}/{sample_folder_name}'
        mark_poly_img_path = f'{mark_yolo_folder}/runs/segment/predict/{all_sample_folder_name}/{sample_folder_name}'
        generate_verify_video(all_2D_centers, ball_bbox_img_path, mark_poly_img_path, 
                              output_path= f'{output_sample_folder_path}/verify_video.mp4')

    # ----------------------------------------------------------------
    # 計算3D座標
    # ----------------------------------------------------------------
    if CALCULATE_3D:
        camParams = read_calibration_file(camParamsPath)
        lb, rb, lmo, rmo, lmx, rmx = extract_centers(all_2D_centers, total_frames=500)

        left_corners = np.loadtxt(f'{output_folder_path}/left_corners.txt')
        right_corners = np.loadtxt(f'{output_folder_path}/right_corners.txt')
        
        print('🚀 計算球桌角落3D座標...')
        corners_3D, _, _ = myDLT(camParams, left_corners, right_corners)
        print('🚀 計算軌跡3D座標...')
        traj_3D, traj_reproj_errors_L, traj_reproj_errors_R = myDLT(camParams, lb, rb)

        # 根據球心座標和球面方程式計算標記3D座標
        marks_3D, m_reproj_errors_L, m_reproj_errors_R = get_marks_3D(camParams, traj_3D, lmo, rmo, lmx, rmx, 
                                                                      output_dir=f"{output_sample_folder_path}/marks_intersection")    
        # 輸出重投影誤差圖表
        plot_reprojection_error(
            traj_reproj_errors_L, traj_reproj_errors_R,
            m_reproj_errors_L, m_reproj_errors_R,
            path = f'{output_sample_folder_path}/reprojection_errors.jpg'
        )

        # 轉換為自訂的坐標系
        corners_3D_transformed, _ = transform_coord_system(corners_3D, corners_3D)
        traj_3D_transformed, _ = transform_coord_system(traj_3D, corners_3D)
        marks_3D_transformed = shift_marks_by_trajectory(traj_3D, traj_3D_transformed, marks_3D)
        # marks_3D_transformed, _ = transform_coord_system(marks_3D, corners_3D)

        np.savetxt(f'{output_folder_path}/corners_3D_transformed.txt', corners_3D_transformed)
        np.savetxt(f'{output_sample_folder_path}/traj_3D_transformed.txt', traj_3D_transformed)
        np.savetxt(f'{output_sample_folder_path}/marks_3D_transformed.txt', marks_3D_transformed)

        plot_multiple_3d_trajectories_with_plane([traj_3D_transformed], [marks_3D_transformed], corners_3D_transformed, 
                                                 rotation_axis_list=None, 
                                                 output_html=f'{output_sample_folder_path}/traj_ori.html')

        
        # 移除軌跡異常點 平滑軌跡 標記點隨平滑後的軌跡平移
        cleaned_traj, outlier_idx = remove_outliers_by_knn_distance(traj_3D_transformed, k=5, sigma_thres=3.0)

        # 找出包含軌跡的 frame 和 start_idx, end_idx 從頭尾檢查非空值
        cleaned_traj, start_idx, end_idx = extract_valid_trajectory(cleaned_traj)
        marks_3D_transformed = marks_3D_transformed[start_idx:end_idx+1]

        # 偵測碰撞點 並根據碰撞點切分軌跡和標記
        temp_smoothed_traj = kalman_smooth_with_interp(cleaned_traj, smooth_strength=2, extend_points=10, dt=dt)     # 暫時平滑軌跡 有助找出碰傳idx
        collisions = detect_table_tennis_collisions(temp_smoothed_traj, corners_3D_transformed, z_tolerance=500)

        # print(collisions)
        # collisions[0] = (100, temp_smoothed_traj[100])

        traj_3D_segs = split_trajectory_by_collisions(cleaned_traj, collisions)
        marks_3D_segs = split_trajectory_by_collisions(marks_3D_transformed, collisions)

        # 切分後每段軌跡分開平滑
        for i in range(len(traj_3D_segs)):
            smoothed_traj_seg = kalman_smooth_with_interp(traj_3D_segs[i], smooth_strength=2, extend_points=10, dt=dt)
            marks_3D_segs[i] = shift_marks_by_trajectory(traj_3D_segs[i], smoothed_traj_seg, marks_3D_segs[i])
            traj_3D_segs[i] = smoothed_traj_seg
            np.savetxt(f'{output_sample_folder_path}/smoothed_traj{i+1}.txt', traj_3D_segs[i])

    # ----------------------------------------------------------------
    # 計算旋轉速度
    # ----------------------------------------------------------------
    if CALCULATE_SPIN_RATE:

        # =======================================
        # 用標記位置計算轉速和旋轉軸
        # =======================================
        spin_axis_list = []  
        corners_3D_transformed_meter = corners_3D_transformed / 1000
        for i in range(len(traj_3D_segs)):

            offsets = marks_3D_segs[i] - traj_3D_segs[i]        # 計算標記相對球心的位置
            plane, filtered_offsets = ransac_fit_plane(offsets, iterations=100, threshold=5)
            # plane, filtered_offsets = fit_plane_with_prior(offsets, lam=500)
            spin_axis = plane['normal']
            spin_axis_list.append(spin_axis)
            print(spin_axis)

            # 刪除和旋轉軸偏差過大的標記點
            for j in range(len(filtered_offsets)):
                if np.isnan(filtered_offsets[j][0]):
                    marks_3D_segs[i][j] = np.array([np.nan, np.nan, np.nan])

            if not np.isnan(spin_axis[0]):
                plot_spin_axis_with_fit_plane(offsets, filtered_offsets, plane, 
                                              path=f"{output_sample_folder_path}/spin_axis_seg{i+1}.html")

        # 畫軌跡、標記、旋轉軸(分段用不同顏色區分)
        plot_multiple_3d_trajectories_with_plane(traj_3D_segs, marks_3D_segs, corners_3D_transformed, 
                                                 rotation_axis_list=spin_axis_list, 
                                                 output_html=f'{output_sample_folder_path}/traj_segs.html')

        # 對每條軌跡計算後選角速度
        candidate_rounds = ["CW", "CW_EXTRA", "CCW", "CCW_EXTRA"]
        with open(f"{output_sample_folder_path}/spin_calculation_results.txt", "w") as f:
            for i in range(len(traj_3D_segs)):

                spin_axis = spin_axis_list[i]
                if np.isnan(spin_axis[0]):      # 如果沒有足夠的標記座標(至少三個)可以擬和平面 跳過後續轉速計算
                    continue
                
                candidate_rps_lists, valid_mark_frames = calc_candidate_spin_rates(traj_3D_segs[i], marks_3D_segs[i], spin_axis, fps=FPS)

                if valid_mark_frames == []:     # 如果沒有任何連續的frame 偵測到標記
                    continue

                theta_degs = [frame[-1] for frame in valid_mark_frames]
                
                marks_count = np.sum(~np.isnan(marks_3D_segs[i]).all(axis=1))
                traj_count = np.sum(~np.isnan(traj_3D_segs[i]).all(axis=1))
                mark_detect_rate = round(marks_count / traj_count, 2)

                plot_projected_marks_on_plane_all_frame(valid_mark_frames, spin_axis, 
                                                        save_html=f"{output_sample_folder_path}/spin_animation_{i+1}.html")

                traj_3D_segs[i] /= 1000    # mm轉為公尺

                # 計算平均速度
                displacements = np.diff(traj_3D_segs[i], axis=0)
                distances = np.linalg.norm(displacements, axis=1)
                v_avg = round(np.mean(distances / dt), 4)

                candidate_trajectories = []
                results = []

                # 計算四種旋轉速度回推的軌跡
                for j, candidate_rps_list in enumerate(candidate_rps_lists):
                    best_rps = find_best_rps(candidate_rps_list)
                    candidate_traj = compute_trajectory_continuous(traj_3D_segs[i], dt, FPS, aero_params,
                                                                   best_rps, spin_axis)
                    candidate_trajectories.append(candidate_traj)
                    results.append([candidate_rounds[j], spin_axis, best_rps, candidate_traj])

                f.write(f"\n======= Trajectory Segment {i+1} =======\n")
                f.write(f"Mark detection successful rate: {mark_detect_rate}\n")
                f.write(f"Valid mark frame count: {len(valid_mark_frames)}\n")
                f.write(f"Average Speed: {v_avg} m/s\n")
                f.write("===================================\n")
                for j in range(len(results)):
                    candidate_round, spin_axis, best_rps, candidate_traj = results[j]

                    f.write(f"Candidate_{candidate_round}: \n")
                    f.write(f"Spin Axis: {list(map(lambda x: round(x, 4), spin_axis))}\n")
                    f.write(f"Spin Rate: {round(best_rps, 4)} RPS\n")
                    f.write("===================================\n")
                
                plot_candidate_trajectories(traj_3D_segs[i], candidate_trajectories, spin_axis, corners_3D_transformed_meter,
                                            f"{output_sample_folder_path}/candidate_trajectories_{i+1}.html")
