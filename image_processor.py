import cv2
import os
import numpy as np
from tqdm import tqdm
from calculation_3D import extract_centers

# 分割為左右影像
def split_image(image):
    height, width = image.shape[:2]
    half_width = width // 2  # 影像寬度一半
    left_image = image[:, :half_width]  # 左半部
    right_image = image[:, half_width:]  # 右半部
    return left_image, right_image

# 增強影像
def enhance_image(image, alpha=1.5, beta=30):
    enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)  # 調整亮度與對比度：image * alpha + beta
    return enhanced

# 生成逐幀驗證影片
def generate_verify_video(all_2D_centers, ball_bbox_img_path, mark_poly_img_path, output_path, 
                          fps=30, total_frames=500, frame_width=1440, frame_height=1080, ignore_rate=0.05):
    
    display_width, display_height = frame_width // 2, frame_height // 2  # 720x540
    video_width = display_width * 2  # 1440
    video_height = display_height  # 540
    small_img_size = (frame_width // 10, frame_width // 10)  # 144x144

    # 初始化 VideoWriter
    video_writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (video_width, video_height)
    )

    print(f'🚀 輸出影片: {output_path} ')
    for frame_num in tqdm(range(total_frames)):
        frame_str = f"{frame_num:04d}"
        name_L = f"image-{frame_str}_L.jpg"
        name_R = f"image-{frame_str}_R.jpg"

        # 讀取主圖並縮小
        path_L = os.path.join(ball_bbox_img_path, name_L)
        path_R = os.path.join(ball_bbox_img_path, name_R)

        img_L = cv2.imread(path_L)
        img_R = cv2.imread(path_R)

        if img_L is None or img_R is None:
            print(f"Frame {frame_str} not found in ball_bbox_label_path.")
            continue

        img_L = cv2.resize(img_L, (display_width, display_height))
        img_R = cv2.resize(img_R, (display_width, display_height))

        combined_img = np.hstack((img_L, img_R))

        # 加入 frame number（白字、粗體、置中頂部）
        cv2.putText(
            combined_img,
            frame_str,
            (video_width // 2 - 50, 30),  # 大約置中
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # 小圖：左上角與右上角
        small_L_path = os.path.join(mark_poly_img_path, name_L)
        small_R_path = os.path.join(mark_poly_img_path, name_R)

        if os.path.exists(small_L_path) and (name_L.split('.')[0] + '.txt' in all_2D_centers.keys()):
            small_L = cv2.imread(small_L_path)
            small_L = cv2.resize(small_L, small_img_size)
        else:
            small_L = np.zeros((small_img_size[1], small_img_size[0], 3), dtype=np.uint8)

        if os.path.exists(small_R_path) and (name_R.split('.')[0] + '.txt' in all_2D_centers.keys()):
            small_R = cv2.imread(small_R_path)
            small_R = cv2.resize(small_R, small_img_size)
        else:
            small_R = np.zeros((small_img_size[1], small_img_size[0], 3), dtype=np.uint8)

        # 將小圖貼上去
        combined_img[0:small_img_size[1], 0:small_img_size[0]] = small_L
        combined_img[0:small_img_size[1], -small_img_size[0]:] = small_R

        # 寫入影片
        video_writer.write(combined_img)

    video_writer.release()
    print(f"影片輸出完成：{output_path}")

if __name__ == "__main__":
    pass
