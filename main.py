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
CALCULATE_SPIN_RATE = False

all_sample_folder_name = '0415'
sample_folder_name = '20250415_193043'

ori_img_folder_path = os.path.join('CameraControl/bin/x64/TableTennisData/', all_sample_folder_name, sample_folder_name)    # 原影像資料夾路徑
processed_img_folder_path = os.path.join('ProcessedImages', all_sample_folder_name, sample_folder_name)    # 處理後的影像資料夾路徑
os.makedirs(processed_img_folder_path, exist_ok=True) 

ball_yolo_params = {'img_size':640, 'batch':16, 'epochs':100}
mark_yolo_params = {'img_size':128, 'batch':16, 'epochs':100}

output_folder_path = os.path.join('OUTPUT', all_sample_folder_name)
output_sample_folder_path = os.path.join('OUTPUT', all_sample_folder_name, sample_folder_name)
os.makedirs(output_sample_folder_path, exist_ok=True)

camParamsPath = "CameraCalibration/STEREO_IMAGES/cvCalibration_result.txt"

FPS = 225

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

    LR_map = {
                "image-0000": {"L": "image-0000_L.txt", "R": _______None_______},
                "image-0001": {"L": _______None_______, "R": "image-0001_R.txt"},
                "image-0002": {"L": "image-0002_L.txt", "R": "image-0002_R.txt"}
                ...
             }
    """
    
    if CALCULATE_3D:
        camParams = read_calibration_file(camParamsPath)
        # LR_map = create_LR_map(all_2D_centers)
        # lb, rb, lmo, rmo, lmx, rmx = get_LR_centers_with_marks(LR_map, all_2D_centers)

        lb, rb, lmo, rmo, lmx, rmx = extract_centers(all_2D_centers)

        print('🚀 計算3D座標中...')
        left_corners = np.loadtxt(f'{output_folder_path}/left_corners.txt')
        right_corners = np.loadtxt(f'{output_folder_path}/right_corners.txt')
        
        corners_3D = myDLT(camParams, left_corners, right_corners)
        traj_3D = myDLT(camParams, lb, rb)

        # # 軌跡也需記錄掉幀情況
        # print(traj_3D)
        # input()




        # traj_3D_filtered, _ = remove_outliers_by_dbscan(traj_3D, eps=10, min_samples=5)
        # traj_3D_filtered, _ = remove_outliers_by_speed(traj_3D_filtered, max_speed_threshold=30)

        marks_3D = get_marks_3D(camParams, traj_3D, lmo, rmo, lmx, rmx)    # 根據球心座標和球面方程式計算標記3D座標

        # 轉換為自訂的坐標系
        corners_3D_transformed, _ = transform_coord_system(corners_3D, corners_3D)
        traj_3D_transformed, _ = transform_coord_system(traj_3D, corners_3D)
        marks_3D_transformed, _ = transform_coord_system(marks_3D, corners_3D)

        # 套用Kalman Filter做平滑
        traj_3D_transformed_KF = simple_kalman_filter_3d(traj_3D_transformed, FPS)

        # 標記座標跟著平滑後的軌跡一起平移
        diffs = traj_3D_transformed_KF - traj_3D_transformed
        marks_3D_transformed = marks_3D_transformed + diffs

        np.savetxt(f'{output_folder_path}/corners_3D.txt', corners_3D_transformed)
        np.savetxt(f'{output_sample_folder_path}/traj_3D_transformed.txt', traj_3D_transformed_KF)
        np.savetxt(f'{output_sample_folder_path}/marks_3D_transformed.txt', marks_3D_transformed)

        collisions = detect_table_tennis_collisions(traj_3D_transformed_KF, corners_3D_transformed)
        traj_3D_segs = split_trajectory_by_collisions(traj_3D_transformed, collisions)
        marks_3D_segs = split_trajectory_by_collisions(marks_3D_transformed, collisions)

        for i in range(len(traj_3D_segs)):
            traj_list = [traj_3D_segs[i], marks_3D_segs[i]]
            # traj_list = [traj_3D_segs[i], simple_kalman_filter_3d(traj_3D_segs[i], FPS), marks_3D_segs[i]]
            plot_multiple_3d_trajectories_with_plane(traj_list, corners_3D_transformed, f'{output_sample_folder_path}/traj{i+1}.html')

    # ----------------------------------------------------------------
    # Step 8: # 計算旋轉速度
    # ----------------------------------------------------------------
    if CALCULATE_SPIN_RATE:

        # traj_3D = np.loadtxt(f"{output_sample_folder_path}/traj_3D_transformed.txt")
        # marks_3D = np.loadtxt(f"{output_sample_folder_path}/marks_3D_transformed.txt")

        collisions = detect_table_tennis_collisions(traj_3D_transformed_KF, corners_3D_transformed)
        traj_3D_segs = split_trajectory_by_collisions(traj_3D_transformed, collisions)
        marks_3D_segs = split_trajectory_by_collisions(marks_3D_transformed, collisions)

        all_spin_axis = []
        for i in range(len(traj_3D_segs)):
            offsets = calc_offsets(traj_3D_segs[i], marks_3D_segs[i])
            fig, spin_axis = fit_and_plot_offset_plane(offsets)
            all_spin_axis.append(spin_axis)

            spin_axis_graph_path = f"{output_sample_folder_path}/spin_axis_seg{i+1}.html"
            pio.write_html(fig, file=spin_axis_graph_path, auto_open=False)
            print(f"✅ 已輸出至：{spin_axis_graph_path}")

            rps_cw_list, rps_cw_extra_list, rps_ccw_list, rps_ccw_extra_list = calc_candidate_spin_rates(FPS, 
                                                                                                         traj_3D_segs[i], 
                                                                                                         marks_3D_segs[i], 
                                                                                                         spin_axis)
            print(rps_cw_list)
            rps_cw = trimmed_mean_rps(rps_cw_list, trim_frac=0.1)
            print(rps_cw)

            print(rps_cw_extra_list)
            rps_cw_extra = trimmed_mean_rps(rps_cw_extra_list, trim_frac=0.1)
            print(rps_cw_extra)

            print(rps_ccw_list)
            rps_ccw = trimmed_mean_rps(rps_ccw_list, trim_frac=0.1)
            print(rps_ccw)

            print(rps_ccw_extra_list)
            rps_ccw_extra = trimmed_mean_rps(rps_ccw_extra_list, trim_frac=0.1)
            print(rps_ccw_extra)
