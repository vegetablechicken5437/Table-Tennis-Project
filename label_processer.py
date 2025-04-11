import cv2
import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from typing import Tuple

# 讀取 polygon 格式 ===
def read_poly_labels(label_path):
    """
    label_data = [
                    {"class_id": <id1>, "confidence": <conf1>, "polygon": [(x1, y1), (x2, y2), ...]}, 
                    {"class_id": <id2>, "confidence": <conf2>, "polygon": [(x1, y1), (x2, y2), ...]}, 
                    ...
                 ]
    """
    label_data = []
    with open(label_path, 'r') as txt_lines:
        for line in txt_lines:
            parts = list(map(float, line.strip().split()))
            cls_id = int(parts[0])
            conf = parts[-1]
            coords = parts[1:-1]
            polygon = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
            label_data.append({
                "class_id": cls_id,
                "confidence": conf,
                "polygon": polygon  # 相對座標
            })
    return label_data

# 讀取 bbox 格式 ===
def read_bbox_labels(label_path):
    """
    label_data = [
                    {"class_id": <id1>, "confidence": <conf1>, "polygon": [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]}, 
                    {"class_id": <id2>, "confidence": <conf2>, "polygon": [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]}, 
                    ...
                 ]
    """
    label_data = []
    with open(label_path, 'r') as txt_lines:
        for line in txt_lines:
            parts = list(map(float, line.strip().split()))
            cls_id, x_center, y_center, width, height, confidence = parts
            # 轉為四個角點（順時針）
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            polygon = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            label_data.append({
                "class_id": int(cls_id),
                "confidence": confidence,
                "polygon": polygon
            })
    return label_data

# 轉為 pixel 座標 ===
def convert_to_pixel_coords(label_data, image_width, image_height):
    """
    label_data = [
                    {"class_id": <id1>, "confidence": <conf1>, "polygon": <polygon1>, "pixel_polygon": ...}, 
                    {"class_id": <id2>, "confidence": <conf2>, "polygon": <polygon2>, "pixel_polygon": ...}, 
                    ...
                 ]
    """
    for item in label_data:
        item["pixel_polygon"] = [(int(x * image_width), int(y * image_height)) for x, y in item["polygon"]]
    return label_data

# 挑選最大 confidence ===
def select_max_conf_by_class(label_data):
    """
    label_data = [
                    {"class_id": <id1>, "confidence": <conf1>, "polygon": <polygon1>, "pixel_polygon": ..., "center":...}, 
                    {"class_id": <id2>, "confidence": <conf2>, "polygon": <polygon2>, "pixel_polygon": ..., "center":...}, 
                    ...
                 ]
    best_by_class = {
                        cls_id: {"class_id": <id1>, "confidence": <conf1>, "polygon": <polygon1>, "pixel_polygon": ..., "center":...},
                        cls_id: {"class_id": <id1>, "confidence": <conf1>, "polygon": <polygon1>, "pixel_polygon": ..., "center":...},
                        ...
                    }
    """
    best_by_class = {}
    for item in label_data:
        cls_id = item["class_id"]
        conf = item["confidence"]
        if cls_id not in best_by_class or conf > best_by_class[cls_id]["confidence"]:
            best_by_class[cls_id] = item
    return best_by_class

# 計算幾何中心（moments） ===
def compute_polygon_center(label_data):
    """
    label_data = [
                    {"class_id": <id1>, "confidence": <conf1>, "polygon": <polygon1>, "pixel_polygon": ..., "center":...}, 
                    {"class_id": <id2>, "confidence": <conf2>, "polygon": <polygon2>, "pixel_polygon": ..., "center":...}, 
                    ...
                 ]
    """
    for item in label_data:
        contour = np.array(item["pixel_polygon"], dtype=np.int32).reshape((-1, 1, 2))
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            item["center"] = (cx, cy)
        else:
            item["center"] = (None, None)
    return label_data

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

def scale_bounding_box(points, scale):
    # 解包原始座標
    x1, y1 = points[0]
    x2, y2 = points[2]

    # 計算中心點
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    # 計算新的寬高
    width = (x2 - x1) * scale
    height = (y2 - y1) * scale

    # 計算新的 x1, x2, y1, y2
    new_x1 = int(cx - width / 2)
    new_x2 = int(cx + width / 2)
    new_y1 = int(cy - height / 2)
    new_y2 = int(cy + height / 2)

    # 傳回放大後的新座標點
    return [
        (new_x1, new_y1),
        (new_x2, new_y1),
        (new_x2, new_y2),
        (new_x1, new_y2)
    ]

def bbox_edge_constraint(bbox_corners, image_width=1440, image_height=1080):
    x1, y1 = bbox_corners[0]   # 放大後的 bbox xy
    x2, y2 = bbox_corners[2]   # 放大後的 bbox xy
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image_width, x2)
    y2 = min(image_height, y2)
    return (x1, y1, x2, y2)

def crop_bbox(img_folder, ball_bbox_label_path, output_folder, 
              image_width=1440, image_height=1080, scale=2, bbox_width=128, bbox_height=128):
    """
    all_bbox_xyxy = {
                        "label_file1.txt": (x1, y1, x2, y2), 
                        "label_file2.txt": (x1, y1, x2, y2), 
                        ...
                    }
    """
    print('🚀 裁切所有影像的 bbox ...')
    ball_label_files = os.listdir(ball_bbox_label_path)
    filtered_ball_label_files = filter_lr_files(ball_label_files)  # 過濾只保留同時擁有 L 和 R 的檔案

    all_bbox_xyxy = {}
    for label_file_name in tqdm(filtered_ball_label_files):
        label_path = os.path.join(ball_bbox_label_path, label_file_name)    # 讀取影像中的 bbox labels
        ball_label_data = read_bbox_labels(label_path)  
        ball_label_data_pixel_coords = convert_to_pixel_coords(ball_label_data, image_width, image_height)  # 轉換並加入pixel_polygon
        ball_label_data_best_pixel_coords = select_max_conf_by_class(ball_label_data_pixel_coords)    # 篩選最大 conf 的 label
        ball_bbox_corners = ball_label_data_best_pixel_coords[0]["pixel_polygon"]    # [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

        scaled_ball_bbox_corners = scale_bounding_box(ball_bbox_corners, scale)
        x1, y1, x2, y2 = bbox_edge_constraint(scaled_ball_bbox_corners)
        all_bbox_xyxy[label_file_name] = (x1, y1, x2, y2)
        # print((x1, y1, x2, y2))
        
        img_file_name = label_file_name.split('.')[0] + '.jpg'
        img_path = os.path.join(img_folder, img_file_name)
        image = cv2.imread(img_path)

        cropped_ball = image[y1:y2, x1:x2]
        cropped_ball = cv2.resize(cropped_ball, (bbox_width, bbox_height))  # 統一 bbox 尺寸

        output_path = os.path.join(output_folder, img_file_name)
        cv2.imwrite(output_path, cropped_ball)

    print(f"已裁切 {img_folder} 所有 bbox 至 {output_folder}")
    return all_bbox_xyxy

def map_point_back_to_ori_image(point_in_crop, bbox_xyxy, bbox_width=128, bbox_height=128):
    x1, y1, x2, y2 = bbox_xyxy
    bbox_width_ori, bbox_height_ori = abs(int(x2-x1)), abs(int(y2-y1))
    x_in_ori = x1 + point_in_crop[0] * (bbox_width_ori/bbox_width)
    y_in_ori = y1 + point_in_crop[1] * (bbox_height_ori/bbox_height)
    return (x_in_ori, y_in_ori)

def extract_2D_points(mark_poly_label_path, all_bbox_xyxy, bbox_width=128, bbox_height=128):
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
    all_2D_centers = {}
    mark_label_files = os.listdir(mark_poly_label_path)
    for label_file_name in mark_label_files:
        label_path = os.path.join(mark_poly_label_path, label_file_name)
        mark_label_data = read_poly_labels(label_path)
        mark_label_data_pixel_coords = convert_to_pixel_coords(mark_label_data, bbox_width, bbox_height)
        mark_label_data_pixel_coords_centers = compute_polygon_center(mark_label_data_pixel_coords)   # 加入['center']到各label的字典
        mark_label_data_best_pixel_coords_centers = select_max_conf_by_class(mark_label_data_pixel_coords_centers)    # 分別找出最大 conf 的 ball, mark_o, mark_x 的 label
        
        # label_map = {0: "ball", 1: "mark_o", 2: "mark_x"}
        bbox_xyxy = all_bbox_xyxy[label_file_name]
        centers_map = {}
        if 0 in mark_label_data_best_pixel_coords_centers.keys():
            ball_center_in_crop = mark_label_data_best_pixel_coords_centers[0]["center"]
            ball_center = map_point_back_to_ori_image(ball_center_in_crop, bbox_xyxy)
            centers_map[0] = ball_center
        if 1 in mark_label_data_best_pixel_coords_centers.keys():
            mark_o_center_in_crop = mark_label_data_best_pixel_coords_centers[1]["center"]
            mark_o_center = map_point_back_to_ori_image(mark_o_center_in_crop, bbox_xyxy)
            centers_map[1] = mark_o_center
        if 2 in mark_label_data_best_pixel_coords_centers.keys():
            mark_x_center_in_crop = mark_label_data_best_pixel_coords_centers[2]["center"]
            mark_x_center = map_point_back_to_ori_image(mark_x_center_in_crop, bbox_xyxy)
            centers_map[2] = mark_x_center

        all_2D_centers[label_file_name] = centers_map
    return all_2D_centers

def create_LR_map(all_2D_centers):
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
                "image-0000": {"L": "image-0000_L.txt", "R": None},
                "image-0001": {"L": None, "R": "image-0001_R.txt"},
                "image-0002": {"L": "image-0002_L.txt", "R": "image-0002_R.txt"}
             }
    """
    LR_map = {}
    for filename in all_2D_centers.keys():
        if '_L.txt' in filename:
            base = filename.replace('_L.txt', '')
            if base not in LR_map:
                LR_map[base] = {"L": None, "R": None}
            LR_map[base]["L"] = filename
        elif '_R.txt' in filename:
            base = filename.replace('_R.txt', '')
            if base not in LR_map:
                LR_map[base] = {"L": None, "R": None}
            LR_map[base]["R"] = filename

    LR_map = dict(sorted(LR_map.items()))   # 依影像編號排序
    return LR_map

if __name__ == "__main__":
    pass