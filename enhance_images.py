import cv2
import numpy as np
import os
from tqdm import tqdm

def enhance_image(image):
    """
    對影像進行調亮、降噪與增強對比度處理
    """
    bright_image = cv2.convertScaleAbs(image, alpha=1.5, beta=30)  # 調整亮度
    lab = cv2.cvtColor(bright_image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # 調整對比度
    l = clahe.apply(l)
    enhanced_lab = cv2.merge((l, a, b))
    enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return enhanced_image

def enhance_all(ori_img_folder, enhanced_img_folder):
    """對所有影像進行強化

    Args:
        ori_img_folder (_type_): 原影像資料夾
        adjusted_img_folder (_type_): 調整後的輸出資料夾
    """
    os.makedirs(enhanced_img_folder, exist_ok=True)

    print('🚀 增強所有影像中 ...')
    for img_name in tqdm(os.listdir(ori_img_folder)):
        img_path = os.path.join(ori_img_folder, img_name)
        img = cv2.imread(img_path)
        enhanced = enhance_image(img)

        path = os.path.join(enhanced_img_folder, img_name)
        cv2.imwrite(path, enhanced)
    print(f"✅ 所有影像已增強並存至 {enhanced_img_folder}\n")

if __name__ == '__main__':
    ori_img_folder = None
    enhanced_img_folder = None
    enhance_all(ori_img_folder, enhanced_img_folder)
