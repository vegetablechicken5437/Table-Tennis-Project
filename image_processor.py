import cv2
import os
import numpy as np
from tqdm import tqdm

def split_image(image):

    height, width = image.shape[:2]
    half_width = width // 2  # 影像寬度一半
    left_image = image[:, :half_width]  # 左半部
    right_image = image[:, half_width:]  # 右半部

    return left_image, right_image

def enhance_image(image, alpha=1.5, beta=30):

    bright_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)  # 調整亮度
    lab = cv2.cvtColor(bright_image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # 調整對比度
    l = clahe.apply(l)
    enhanced_lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    # gamma = 2
    # invGamma = 1.0 / gamma
    # table = np.array([(i / 255.0) ** invGamma * 255 for i in range(256)]).astype("uint8")
    # enhanced = cv2.LUT(image, table)

    # enhanced = cv2.GaussianBlur(enhanced, (25, 25), 0)
    # enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
    # enhanced = cv2.bilateralFilter(enhanced, 20, 20, 20)
    # enhanced = cv2.medianBlur(enhanced, 5)

    # # Step 3: 銳化
    # sharpen_kernel = np.array([[0, -1, 0],
    #                            [-1, 5, -1],
    #                            [0, -1, 0]])
    # sharpened = cv2.filter2D(enhanced, -1, sharpen_kernel)
    # enhanced = cv2.addWeighted(enhanced, 0.6, sharpened, 0.4, 0)

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

# === 主程式 ===
if __name__ == "__main__":

    img = cv2.imread(r"C:\Users\jason\Desktop\TableTennisProject\CameraControl\0325-2\top5-1\image-0129.jpg")
    enhanced = enhance_image(img, alpha=2, beta=30)
    cv2.imwrite(r"C:\Users\jason\Desktop\TableTennisProject\ProcessedImages\0325-2\top5-1\enhanced\image-0129_EN.jpg", enhanced)

    # image_folder_path = 'samples'
    # output_video_path = 'demo_video.mp4'
    # createVideo(image_folder_path, output_video_path, fps=30)
