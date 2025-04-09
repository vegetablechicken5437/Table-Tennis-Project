import cv2
import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict

def read_yolo_labels(label_path):
    """
    讀取 YOLO 格式的標籤檔
    """
    labels = []
    with open(label_path, "r") as file:
        for line in file:
            parts = line.strip().split()
            cls_id, x_center, y_center, width, height, confidence = map(float, parts)
            labels.append({
                "class_id": int(cls_id),
                "x_center": x_center,
                "y_center": y_center,
                "width": width,
                "height": height,
                "confidence": confidence
            })
    return labels

def filter_lr_files(files):
    """
    過濾只保留同時擁有 L 和 R 的檔案
    :param files: 檔案名稱列表，例如 ['image-0110_L.txt', 'image-0110_R.txt', ...]
    :return: 過濾後的檔案列表
    """
    # 建立字典來記錄 L 和 R 的存在情況
    file_dict = defaultdict(set)

    # 解析數字並記錄是 L 還是 R
    for file in files:
        num_part, lr_part = file.split('_')
        file_dict[num_part].add(lr_part)  # 記錄 L 或 R 是否存在

    # 找出同時擁有 L 和 R 的數字
    valid_numbers = {num for num, sides in file_dict.items() if len(sides) == 2}

    # 只保留屬於 valid_numbers 的檔案
    return [file for file in files if file.split('_')[0] in valid_numbers]

def get_coords_from_bbox(detect_result_path, output_folder, dtype):

    label_img_files = [f for f in os.listdir(detect_result_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    first_img = cv2.imread(os.path.join(detect_result_path, label_img_files[0]))
    H, W = first_img.shape[:2]

    label_files = os.listdir(os.path.join(detect_result_path, 'labels'))
    label_files = filter_lr_files(label_files)  # 過濾只保留同時擁有 L 和 R 的檔案

    left_pts, right_pts = [], []

    for label_file_name in label_files:

        label_path = os.path.join(detect_result_path, 'labels', label_file_name)
        labels = read_yolo_labels(label_path)
        sorted_labels = sorted(labels, key=lambda labels: labels['confidence'], reverse=True)     # 根據confidence排列
        target_label = sorted_labels[0]     # 目標只找最大confidence的label

        # 輸出bbox中心點
        x_center, y_center = int(target_label['x_center'] * W), int(target_label['y_center'] * H)   # 根據影像大小換算座標  
        if 'L' in label_file_name:      # 利用檔名判斷是左邊還是右邊的點
            left_pts.append([x_center, y_center])
        elif 'R' in label_file_name:
            right_pts.append([x_center, y_center])

    if dtype == 'ball':
        left_pts_path = f'{output_folder}/left_balls.txt'
        right_pts_path = f'{output_folder}/right_balls.txt'

        ball_frame_nums = [int(label_file.split('_')[0].split('-')[-1]) for label_file in label_files]
        ball_frame_nums = np.unique(ball_frame_nums)
        ball_frame_nums_path = f'{output_folder}/ball_frame_nums.txt'   # 儲存有偵測到ball的frame number
        np.savetxt(ball_frame_nums_path, ball_frame_nums, fmt="%d")

    elif dtype == 'logo':
        left_pts_path = f'{output_folder}/left_logos.txt'
        right_pts_path = f'{output_folder}/right_logos.txt'

        ball_frame_nums_path = os.path.join(output_folder, 'ball_frame_nums.txt')
        ball_frame_nums = np.loadtxt(ball_frame_nums_path).tolist()
        bbox_info_L_path = os.path.join(output_folder, 'bbox_info_L.txt')
        bbox_info_R_path = os.path.join(output_folder, 'bbox_info_R.txt')
        bbox_info_L, bbox_info_R = np.loadtxt(bbox_info_L_path), np.loadtxt(bbox_info_R_path)

        # bbox_info記錄了所有偵測到的桌球bbox的(x1, y1, x2, y2) 只有LR都有偵測到的才會記錄
        # ball_frame_nums記錄了所有偵測到的桌球bbox的frame number 只有LR都有偵測到的才會記錄
        # left_pts, right_pts記錄了所有偵測到的logo的x_center, y_center

        logo_frame_nums = [int(label_file.split('_')[0].split('-')[-1]) for label_file in label_files]
        logo_frame_nums = np.unique(logo_frame_nums)
        logo_frame_nums_path = f'{output_folder}/logo_frame_nums.txt'   # 儲存有偵測到logo的frame number
        np.savetxt(logo_frame_nums_path, logo_frame_nums, fmt="%d")

        for i, left_pt in enumerate(left_pts):
            idx = ball_frame_nums.index(logo_frame_nums[i])
            left_pts[i] = bbox_info_L[idx][:2] + left_pt

        for i, right_pt in enumerate(right_pts):
            idx = ball_frame_nums.index(logo_frame_nums[i])
            right_pts[i] = bbox_info_R[idx][:2] + right_pt

    np.savetxt(left_pts_path, np.array(left_pts), fmt="%.4f")   # 儲存桌球2D座標
    np.savetxt(right_pts_path, np.array(right_pts), fmt="%.4f")
    print(f'✅ 左右影像的 {dtype} 2D 座標已儲存於 {left_pts_path}, {right_pts_path}')

    return left_pts, right_pts

def crop_ball_from_image(image, label, image_width, image_height, scale_factor=2, size_x=128):
    """
    根據 YOLO 標籤裁切桌球影像，並放大 bbox
    """
    x_center = int(label["x_center"] * image_width)
    y_center = int(label["y_center"] * image_height)
    box_width = int(label["width"] * image_width * scale_factor)
    box_height = int(label["height"] * image_height * scale_factor)

    x1 = max(0, x_center - box_width // 2)
    y1 = max(0, y_center - box_height // 2)
    x2 = min(image_width, x_center + box_width // 2)
    y2 = min(image_height, y_center + box_height // 2)

    cropped_ball = image[y1:y2, x1:x2]
    # cropped_ball = cv2.resize(cropped_ball, size, interpolation=cv2.INTER_AREA)
    cropped_ball = cv2.resize(cropped_ball, (size_x, int(cropped_ball.shape[0] * (size_x / cropped_ball.shape[1]))))

    return cropped_ball, x1, y1, x2, y2

def crop_bbox(img_folder, detect_result_path, output_folder):
    """
    處理多個資料夾中的影像，裁切桌球並增強影像，根據 x_center 判斷左半邊或右半邊，並只保留 confidence 最高的標籤
    """
    img_files = [f for f in os.listdir(img_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    first_img = cv2.imread(os.path.join(img_folder, img_files[0]))
    H, W = first_img.shape[:2]

    label_files = os.listdir(os.path.join(detect_result_path, 'labels'))
    label_files = filter_lr_files(label_files)  # 過濾只保留同時擁有 L 和 R 的檔案

    bbox_info_L, bbox_info_R = [], []
    print('🚀 裁切所有影像的 bbox ...')
    for label_file_name in tqdm(label_files):
        label_path = os.path.join(detect_result_path, 'labels', label_file_name)
        labels = read_yolo_labels(label_path)
        sorted_labels = sorted(labels, key=lambda labels: labels['confidence'], reverse=True)     # 根據confidence排列
        target_label = sorted_labels[0]     # 目標只找最大confidence的label

        img_file_name = label_file_name.split('.')[0] + '.jpg'
        img_path = os.path.join(img_folder, img_file_name)
        image = cv2.imread(img_path)

        cropped_ball, x1, y1, x2, y2 = crop_ball_from_image(image, target_label, W, H)
        if 'L' in label_file_name:
            bbox_info_L.append([x1, y1, x2, y2])
        elif 'R' in label_file_name:
            bbox_info_R.append([x1, y1, x2, y2])

        output_path = os.path.join(output_folder, img_file_name)
        cv2.imwrite(output_path, cropped_ball)

    print(f"已裁切 {img_folder} 所有bbox ，輸出至 {output_folder}")
    return np.array(bbox_info_L), np.array(bbox_info_R)

if __name__ == "__main__":
    pass