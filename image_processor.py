import cv2
import os
import numpy as np
from tqdm import tqdm
from calculation_3D import extract_centers

def split_image(image):

    height, width = image.shape[:2]
    half_width = width // 2  # 影像寬度一半
    left_image = image[:, :half_width]  # 左半部
    right_image = image[:, half_width:]  # 右半部

    return left_image, right_image

# def adjust_cropped_ball(image):
#     return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 3, 11)

def enhance_image(image, alpha=1.5, beta=30):

    enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)  # 調整亮度

    # lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
    # l, a, b = cv2.split(lab)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # 調整對比度
    # l = clahe.apply(l)
    # enhanced_lab = cv2.merge((l, a, b))
    # enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    # gamma = 2
    # invGamma = 1.0 / gamma
    # table = np.array([(i / 255.0) ** invGamma * 255 for i in range(256)]).astype("uint8")
    # enhanced = cv2.LUT(image, table)

    # enhanced = cv2.GaussianBlur(enhanced, (25, 25), 0)
    # enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 3, 11)
    # enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
    # enhanced = cv2.bilateralFilter(enhanced, 20, 20, 20)
    # enhanced = cv2.medianBlur(enhanced, 5)

    # # Step 3: 銳化
    # sharpen_kernel = np.array([[0, -1, 0],
    #                            [-1, 5, -1],
    #                            [0, -1, 0]])
    # sharpened = cv2.filter2D(enhanced, -1, sharpen_kernel)
    # enhanced = cv2.addWeighted(enhanced, 0.6, sharpened, 0.4, 0)

    # enhanced = cv2.GaussianBlur(enhanced, (5, 5), 0)

    return enhanced

def createVideo(image_folder_path, output_video_path, fps=30):
    # 取得圖片檔名並排序
    images = [img for img in os.listdir(image_folder_path) if img.endswith(".jpg") or img.endswith(".png")]
    images.sort()

    # 讀取第一張圖片以取得影片的寬高
    first_image = cv2.imread(os.path.join(image_folder_path, images[0]))
    height, width, layers = first_image.shape

    new_size = (width // 2, height // 2)

    # 定義影片編碼與輸出格式
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 編碼格式
    video = cv2.VideoWriter(output_video_path, fourcc, fps, new_size)

    print(f'🚀 輸出影片: {output_video_path} ')
    # 將每張圖片寫入影片
    for image in tqdm(images):
        img_path = os.path.join(image_folder_path, image)
        frame = cv2.imread(img_path)
        frame = cv2.resize(frame, new_size)
        video.write(frame)

    # 釋放影片寫入器
    video.release()

def generate_verify_video(all_2D_centers, ball_bbox_img_path, mark_poly_img_path, output_path, fps=30, total_frames=500):
    frame_width, frame_height = 1440, 1080
    display_width, display_height = frame_width // 2, frame_height // 2  # 720x540
    video_width = display_width * 2  # 1440
    video_height = display_height  # 540
    small_img_size = (frame_width // 10, frame_width // 10)  # 144x144
    ignore_rate = 0.05

    lb, rb, lmo, rmo, lmx, rmx = extract_centers(all_2D_centers, total_frames=total_frames)

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

        # # 🟨 畫黃色方框
        # box_margin_x = int(display_width * ignore_rate)
        # box_margin_y = int(display_height * ignore_rate)
        # box_width = display_width - 2 * box_margin_x
        # box_height = display_height - 2 * box_margin_y

        # # 左畫面
        # top_left_L = (box_margin_x, box_margin_y)
        # bottom_right_L = (box_margin_x + box_width, box_margin_y + box_height)
        # cv2.rectangle(combined_img, top_left_L, bottom_right_L, (0, 255, 255), 2)

        # # 右畫面
        # offset = display_width
        # top_left_R = (offset + box_margin_x, box_margin_y)
        # bottom_right_R = (offset + box_margin_x + box_width, box_margin_y + box_height)
        # cv2.rectangle(combined_img, top_left_R, bottom_right_R, (0, 255, 255), 2)

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

        # # 🔲 小圖畫黃色方框
        # margin_x_s = small_img_size[0] // 10
        # margin_y_s = small_img_size[1] // 10
        # box_w_s = small_img_size[0] - 2 * margin_x_s
        # box_h_s = small_img_size[1] - 2 * margin_y_s

        # # 左上角小圖框（左上為(0,0)）
        # top_left_s_L = (margin_x_s, margin_y_s)
        # bottom_right_s_L = (margin_x_s + box_w_s, margin_y_s + box_h_s)
        # cv2.rectangle(
        #     combined_img,
        #     top_left_s_L,
        #     bottom_right_s_L,
        #     (0, 255, 255),
        #     2
        # )

        # # 右上角小圖框（從右上 corner 開始算）
        # top_left_s_R = (video_width - small_img_size[0] + margin_x_s, margin_y_s)
        # bottom_right_s_R = (video_width - small_img_size[0] + margin_x_s + box_w_s, margin_y_s + box_h_s)
        # cv2.rectangle(
        #     combined_img,
        #     top_left_s_R,
        #     bottom_right_s_R,
        #     (0, 255, 255),
        #     2
        # )

        # # 🔁 計算縮放比例
        # scale_x = display_width / frame_width  # 720 / 1440 = 0.5
        # scale_y = display_height / frame_height  # 540 / 1080 = 0.5

        # # 🔶 畫上 lb、rb（橘色圓點）
        # if lb[frame_num] is not None:
        #     u, v = lb[frame_num]
        #     u, v = int(u * scale_x), int(v * scale_y)
        #     cv2.circle(combined_img, (u, v), 5, (0, 165, 255), -1)  # 橘色
        # if rb[frame_num] is not None:
        #     u, v = rb[frame_num]
        #     u, v = int(u * scale_x), int(v * scale_y)
        #     cv2.circle(combined_img, (u + display_width, v), 5, (0, 165, 255), -1)  # 右圖 + offset

        # # ⚪ 畫上 lmo、rmo（白色圓點）
        # if lmo[frame_num] is not None:
        #     u, v = lmo[frame_num]
        #     u, v = int(u * scale_x), int(v * scale_y)
        #     cv2.circle(combined_img, (u, v), 5, (255, 255, 255), -1)
        # if rmo[frame_num] is not None:
        #     u, v = rmo[frame_num]
        #     u, v = int(u * scale_x), int(v * scale_y)
        #     cv2.circle(combined_img, (u + display_width, v), 5, (255, 255, 255), -1)

        # # ❌ 畫上 lmx、rmx（白色叉叉）
        # if lmx[frame_num] is not None:
        #     u, v = lmx[frame_num]
        #     u, v = int(u * scale_x), int(v * scale_y)
        #     cv2.line(combined_img, (u - 5, v - 5), (u + 5, v + 5), (255, 255, 255), 2)
        #     cv2.line(combined_img, (u - 5, v + 5), (u + 5, v - 5), (255, 255, 255), 2)
        # if rmx[frame_num] is not None:
        #     u, v = rmx[frame_num]
        #     u, v = int(u * scale_x), int(v * scale_y)
        #     u += display_width
        #     cv2.line(combined_img, (u - 5, v - 5), (u + 5, v + 5), (255, 255, 255), 2)
        #     cv2.line(combined_img, (u - 5, v + 5), (u + 5, v - 5), (255, 255, 255), 2)

        # 寫入影片
        video_writer.write(combined_img)

    video_writer.release()
    print(f"影片輸出完成：{output_path}")

# === 主程式 ===
if __name__ == "__main__":

    # ori_img_folder_path = 'TEMP'
    # output_folder_path = 'TEMP_EN'

    # for i, image_file_name in enumerate(tqdm(os.listdir(ori_img_folder_path))):
    #     image_path = os.path.join(ori_img_folder_path, image_file_name)
    #     img = cv2.imread(image_path)
    #     enhanced = enhance_image(img, 2, 30)
    #     cv2.imwrite(f"{output_folder_path}/{i}.jpg", enhanced)

    # image_folder_path = r"C:\Users\jason\Desktop\TableTennisProject\ProcessedImages\0412\20250412_152611\enhanced_R"
    # output_video_path = 'demo_video_R.mp4'
    # createVideo(image_folder_path, output_video_path, fps=30)

    generate_verify_video(
        ball_bbox_label_path=r"C:\Users\jason\Desktop\TableTennisProject\BallDetection_YOLOv5\yolov5\runs\detect\0412\exp_20250412_152611",
        mark_poly_label_path=r"C:\Users\jason\Desktop\TableTennisProject\LogoDetection_YOLOv8\runs\segment\predict\0412\20250412_152611",
        output_path='output_video.mp4'
    )

    # # === 設定影片路徑 ===
    # input_video_path = 'demo_video_R.mp4'    # 輸入影片路徑
    # output_mask_path = 'output_mask_R.mp4'  # 輸出遮罩影片

    # # === 建立背景相減器 ===
    # backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

    # # === 開啟影片檔案 ===
    # cap = cv2.VideoCapture(input_video_path)
    # if not cap.isOpened():
    #     print("❌ 無法開啟影片")
    #     exit()

    # # === 擷取影片資訊（幀率與大小） ===
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # # === 建立影片寫入器 ===
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter(output_mask_path, fourcc, fps, (width, height), isColor=False)

    # # === 逐幀處理 ===
    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break

    #     # 前景遮罩擷取
    #     fgMask = backSub.apply(frame)

    #     # 寫入黑白遮罩畫面
    #     out.write(fgMask)

    #     # 可視化（開發時可用）
    #     # cv2.imshow('Mask', fgMask)
    #     # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     #     break

    # # === 清理資源 ===
    # cap.release()
    # out.release()
    # cv2.destroyAllWindows()
    # print("✅ 處理完成，遮罩影片已輸出到:", output_mask_path)
