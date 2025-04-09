import os
import cv2
import numpy as np
from image_processor import *
from label_processer import *
from yolo_runner import *
from pick_corners import CornerPicker
import calculation_3D as calc3D
from scipy.ndimage import gaussian_filter1d
from spin_calculation import fit_spin_axis, calc_candidate_spin_rates, compute_trajectory_aero, find_best_matching_rps
from visualize_functions import draw_trajectories, plot_trajectory_with_spin, draw_spin_axis

PROCESS_IMAGE = False
CREATE_VIDEO = False
TRAIN = {'Ball':False, 'Logo':False}
INFERENCE = {'Ball':False, 'Logo':True}
CROP_BBOX = False
EXTRACT_2D_POINTS = False
PICK_CORNERS = False
CALCULATE_3D = False
CALCULATE_SPIN_RATE = False

all_sample_folder_name = '0401'
sample_folder_name = '20250401_203921'

ori_img_folder_path = os.path.join('CameraControl/bin/x64/TableTennisData/', all_sample_folder_name, sample_folder_name)    # 原影像資料夾路徑
processed_img_folder_path = os.path.join('ProcessedImages', all_sample_folder_name, sample_folder_name)    # 處理後的影像資料夾路徑
os.makedirs(processed_img_folder_path, exist_ok=True) 

ball_yolo_params = {'img_size':640, 'batch':16, 'epochs':50}
mark_yolo_params = {'img_size':128, 'batch':16, 'epochs':100}

output_folder_path = os.path.join('OUTPUT', all_sample_folder_name, sample_folder_name)
os.makedirs(output_folder_path, exist_ok=True)

camParamsPath = "CameraCalibration/STEREO_IMAGES/cvCalibration_result.txt"

fps = 225

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
            # img = cv2.bilateralFilter(img, 5, 0, 0)
            imgL, imgR = split_image(enhanced)

            # cv2.imwrite(os.path.join(processed_folder_path, f"{os.path.splitext(image_file_name)[0]}_EN.jpg"), enhanced)
            cv2.imwrite(os.path.join(processed_folder_path, f"{os.path.splitext(image_file_name)[0]}_L.jpg"), imgL)
            cv2.imwrite(os.path.join(processed_folder_path, f"{os.path.splitext(image_file_name)[0]}_R.jpg"), imgR)
            cv2.imwrite(os.path.join(processed_L_folder_path, f"{os.path.splitext(image_file_name)[0]}_L.jpg"), imgL)
            cv2.imwrite(os.path.join(processed_R_folder_path, f"{os.path.splitext(image_file_name)[0]}_R.jpg"), imgR)

    if CREATE_VIDEO:
        for folder_path in (processed_L_folder_path, processed_R_folder_path):
            createVideo(folder_path, f'{folder_path.split('/')[-1]}.mp4', fps=20)

    # ----------------------------------------------------------------
    # Step 3: YOLO偵測桌球(可選擇是否訓練和預測)
    # ----------------------------------------------------------------
    ball_yolo_folder = 'BallDetection_YOLOv5/yolov5'
    # ball_detection_yolov5(ball_yolo_folder, ball_yolo_params, ori_img_folder_path, all_sample_folder_name, sample_folder_name, TRAIN, INFERENCE)
    ball_detection_yolov5(ball_yolo_folder, ball_yolo_params, processed_folder_path, all_sample_folder_name, sample_folder_name, TRAIN, INFERENCE)
    # ----------------------------------------------------------------

    # ----------------------------------------------------------------
    # Step 4: 裁切bounding box 並輸出裁切圖片
    # ----------------------------------------------------------------
    ball_bbox_label_path = os.path.join(ball_yolo_folder, f'runs/detect/{all_sample_folder_name}/exp_{sample_folder_name}/labels')    # 偵測結果資料夾(含多條軌跡的偵測結果)
    cropped_balls_folder = os.path.join('Cropped_Balls', all_sample_folder_name, sample_folder_name)
    os.makedirs(cropped_balls_folder, exist_ok=True)
    
    if CROP_BBOX:
        all_bbox_xyxy = crop_bbox(processed_folder_path, ball_bbox_label_path, cropped_balls_folder)

    # ----------------------------------------------------------------
    # Step 5: YOLO偵測Logo(可選擇是否訓練和預測)
    # ----------------------------------------------------------------
    mark_yolo_folder = 'LogoDetection_YOLOv8'
    logo_detection_yolov8(mark_yolo_folder, mark_yolo_params, cropped_balls_folder, all_sample_folder_name, sample_folder_name, TRAIN, INFERENCE)

    # ----------------------------------------------------------------
    # Step 6: 輸出球和logo在影像上的座標(每個frame都有左、右影像的球座標)
    # ----------------------------------------------------------------
    mark_poly_label_path = f'{mark_yolo_folder}/runs/segment/predict/{all_sample_folder_name}/{sample_folder_name}/labels'
    if EXTRACT_2D_POINTS:
        all_2D_centers = extract_2D_points(mark_poly_label_path, all_bbox_xyxy)

    LR_map = create_LR_map(all_2D_centers)
    # ----------------------------------------------------------------
    
    # ----------------------------------------------------------------
    # Step 7: # 透過UI介面手動選取球桌四角，定義世界坐標系
    # ----------------------------------------------------------------
    if PICK_CORNERS:
        picker = CornerPicker([], output_folder_path)
        picker.pick_corners(processed_folder_path)
        left_corners, right_corners = picker.left_corners, picker.right_corners
    # ----------------------------------------------------------------

    # ----------------------------------------------------------------
    # Step 7: # 計算3D座標
    # ----------------------------------------------------------------
    if CALCULATE_3D:
        camParams = calc3D.read_calibration_file(camParamsPath)

        os.chdir(output_folder_path)
        left_corners, right_corners = np.loadtxt('left_corners.txt'), np.loadtxt('right_corners.txt')
        left_balls, right_balls = np.loadtxt('left_balls.txt'), np.loadtxt('right_balls.txt')
        left_logos, right_logos = np.loadtxt('left_logos.txt'), np.loadtxt('right_logos.txt')
        os.chdir('..')
        os.chdir('..')



        print('🚀 計算3D座標中...')
        corners_3D = calc3D.myDLT(camParams, left_corners, right_corners)

        traj_3D = calc3D.myDLT(camParams, left_balls, right_balls)
        # traj_3D = gaussian_filter1d(traj_3D, sigma=2, axis=0)
        logos_3D = calc3D.myDLT(camParams, left_logos, right_logos)  # logo 2D座標是根據裁切影像
        
        corners_3D_path = f'{output_folder_path}/corners_3D.txt'
        # np.savetxt(corners_3D_path, corners_3D)
        calc3D.changeCoordSys(corners_3D, corners_3D, corners_3D_path)

        traj_3D_path = f'{output_folder_path}/traj_3D.txt'
        calc3D.changeCoordSys(corners_3D, traj_3D, traj_3D_path)
        
        logos_3D_path = f'{output_folder_path}/logos_3D.txt'
        calc3D.changeCoordSys(corners_3D, logos_3D, logos_3D_path)

    # ----------------------------------------------------------------
    # Step 8: # 計算旋轉速度
    # ----------------------------------------------------------------
    if CALCULATE_SPIN_RATE:
        ball_frame_nums = np.loadtxt(f'{output_folder_path}/ball_frame_nums.txt', dtype=int)
        logo_frame_nums = np.loadtxt(f'{output_folder_path}/logo_frame_nums.txt', dtype=int)
        traj_3D = np.loadtxt(f'{output_folder_path}/traj_3D.txt')  # 完整球軌跡
        logos_3D = np.loadtxt(f'{output_folder_path}/logos_3D.txt')  # 偵測到logo的3D點

        if isinstance(logo_frame_nums, np.ndarray) and logo_frame_nums.size > 2:

            # 計算旋轉軸
            translated_logos, plane_normal = fit_spin_axis(ball_frame_nums, logo_frame_nums, traj_3D, logos_3D)

            # draw_spin_axis(translated_logos, plane_normal)

            # # 計算每一幀的速度 (忽略最後一幀)
            # velocity = np.diff(traj_3D, axis=0, prepend=traj_3D[0].reshape(1, -1))
            # # 計算旋轉軸的方向 (根據球的運動)
            # rotation_axis = np.cross(velocity, plane_normal)
            # # 確保法向量朝向正確
            # if np.mean(np.dot(rotation_axis, plane_normal)) < 0:
            #     plane_normal = -plane_normal  # 反轉法向量方向

            # 計算可能的四種角速度
            rps_cw, rps_cw_extra, rps_ccw, rps_ccw_extra = calc_candidate_spin_rates(ball_frame_nums, logo_frame_nums, traj_3D, logos_3D, plane_normal)

            print("\nRotation Axis (Plane Normal):", plane_normal)
            print(f'CW: {rps_cw} rps')
            print(f'CW_extra: {rps_cw_extra} rps')
            print(f'CCW: {rps_ccw} rps')
            print(f'CW_extra: {rps_ccw_extra} rps\n')

            # 空氣動力學參數: [重力加速度 (m/s^2), 桌球質量 (kg), 空氣密度 (kg/m^3), 球的迎風面積 (m^2), 球半徑 (m), 阻力係數, 馬格努斯力係數]
            aero_params = {'g':9.8, 'm':0.0027, 'rho':1.2, 'A':0.001256, 'r':0.02, 'Cd':0.5, 'Cm':1.23}

            # 計算每一幀的速度 (Ground Truth)
            dt_list = np.diff(ball_frame_nums) / fps  # 每一幀的時間間隔 (秒)
            velocity_gt = np.diff(traj_3D, axis=0) * fps  # 速度計算
            acceleration_gt = np.diff(velocity_gt, axis=0) * fps  # 加速度計算

            # 設定模擬步數
            num_steps = len(traj_3D)

            # 計算四種旋轉條件的軌跡
            trajectory_cw = compute_trajectory_aero(velocity_gt[0], traj_3D[0], rps_cw, dt_list, num_steps, plane_normal, aero_params)
            trajectory_cw_extra = compute_trajectory_aero(velocity_gt[0], traj_3D[0], rps_cw_extra, dt_list, num_steps, plane_normal, aero_params)
            trajectory_ccw = compute_trajectory_aero(velocity_gt[0], traj_3D[0], rps_ccw, dt_list, num_steps, plane_normal, aero_params)
            trajectory_ccw_extra = compute_trajectory_aero(velocity_gt[0], traj_3D[0], rps_ccw_extra, dt_list, num_steps, plane_normal, aero_params)
            
            # 畫出可能的軌跡
            candidate_trajectories_path = f'{output_folder_path}/candidate_trajectories.html'
            draw_trajectories(traj_3D, trajectory_cw, trajectory_cw_extra, trajectory_ccw, trajectory_ccw_extra, candidate_trajectories_path)
            
            # 建立字典
            rps_dict = {"cw": rps_cw, "cw_extra": rps_cw_extra, "ccw": rps_ccw, "ccw_extra": rps_ccw_extra}
            aero_trajectories = {"cw": trajectory_cw, "cw_extra": trajectory_cw_extra, "ccw": trajectory_ccw, "ccw_extra": trajectory_ccw_extra}

            # 比對軌跡並找出最佳轉速
            best_rps = find_best_matching_rps(traj_3D, aero_trajectories, rps_dict)
            print(f'Best RPS: {best_rps} rps')
            np.savetxt(f'{output_folder_path}/best_rps.txt', [best_rps])

            # 畫出最終軌跡和轉速
            traj_with_spin_path = f"{output_folder_path}/trajectory_with_spin.html"
            plot_trajectory_with_spin(traj_3D, plane_normal, best_rps, traj_with_spin_path)