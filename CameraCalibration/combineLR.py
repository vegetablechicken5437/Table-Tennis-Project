import os
import cv2
import numpy as np

LR_img_path = r"C:\Users\jason\Desktop\Lab_Report\CameraCalibration\LR_IMAGES"
stereo_img_path = r"C:\Users\jason\Desktop\Lab_Report\CameraCalibration\STEREO_IMAGES"

L_imgs = []
R_imgs = []
for file_name in os.listdir(LR_img_path):
    img_path = os.path.join(LR_img_path, file_name)
    img = cv2.imread(img_path)
    if 'L' in file_name:
        L_imgs.append(img)
    else:
        R_imgs.append(img)

H, W = L_imgs[0].shape[:2]

for i in range(len(L_imgs)):
    res = np.zeros((H, W*2, 3))
    res[:, :W] = L_imgs[i]
    res[:, W:] = R_imgs[i]
    img_path = os.path.join(stereo_img_path, f'{i}.bmp')
    cv2.imwrite(img_path, res)
