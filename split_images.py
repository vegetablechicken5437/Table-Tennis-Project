import cv2
import os
from tqdm import tqdm

def split_left_right_images(sample_folder, sample_folder_LR):
    """
    讀取 `Images/` 內的所有影像，將左右影像分開並儲存到 `ImagesLR/`
    """
    input_dir, output_dir = sample_folder, sample_folder_LR
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(input_dir):
        print(f"❌ 來源資料夾不存在: {input_dir}")
        return

    print('🚀 分割所有影像中 ...')
    for image_file in tqdm(os.listdir(input_dir)):
        image_path = os.path.join(input_dir, image_file)
        output_path_L = os.path.join(output_dir, f"{os.path.splitext(image_file)[0]}_L.jpg")
        output_path_R = os.path.join(output_dir, f"{os.path.splitext(image_file)[0]}_R.jpg")

        # 讀取影像
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ 無法讀取影像: {image_path}")
            continue

        height, width = image.shape[:2]
        half_width = width // 2  # 影像寬度一半

        # 分割左右影像
        left_image = image[:, :half_width]  # 左半部
        right_image = image[:, half_width:]  # 右半部

        # 儲存左右影像
        cv2.imwrite(output_path_L, left_image)
        cv2.imwrite(output_path_R, right_image)

        # print(f"✅ 已分割: {image_file} → 左右影像儲存至 {output_dir}")

    print(f"✅ 所有影像已分割並存至 {output_dir}\n")

# === 主程式 ===
if __name__ == "__main__":
    all_img_folder = 'Images'
    all_img_folder_LR = 'Images_LR'
    sample_folder_name = 'sample-1x'
    sample_folder = f'{all_img_folder}/{sample_folder_name}'            # ex: Images/sample-1
    sample_folder_LR = f'{all_img_folder_LR}/{sample_folder_name}_LR'   # ex: Images_LR/sample-1_LR
    if not os.path.exists(sample_folder_LR):                            # 若分割影像資料夾不在
        split_left_right_images(sample_folder, sample_folder_LR)
