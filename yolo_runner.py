import os
from glob import glob
from ultralytics import YOLO

def ball_detection_yolov5(ball_yolo_folder, ball_yolo_params, img_folder, sample_folder_name, TRAIN, INFERENCE):
    train_py = os.path.join(ball_yolo_folder, 'train.py')
    detect_py = os.path.join(ball_yolo_folder, 'detect.py')
    train_result_folder = os.path.join(ball_yolo_folder, 'runs/train')

    if TRAIN['Ball'] or not os.path.exists(train_result_folder):
        os.system(f'python {train_py} --img {ball_yolo_params["img_size"]} --batch {ball_yolo_params["batch"]} --epochs {ball_yolo_params["epochs"]} --data {ball_yolo_folder}/data.yaml --weights {ball_yolo_folder}/yolov5s.pt')
    
    weights_path = os.path.join(train_result_folder, 'exp/weights/best.pt')
    if INFERENCE['Ball']:
        os.system(f'python {detect_py} --weights {weights_path} --img {ball_yolo_params['img_size']} --source {img_folder} --save-txt --save-conf --project {ball_yolo_folder}/runs/detect --name exp_{sample_folder_name} --exist-ok')

def logo_detection_yolov8(logo_yolo_folder, logo_yolo_params, img_folder, sample_folder_name, TRAIN, INFERENCE):

    os.chdir(logo_yolo_folder)
    detect_folder = 'runs/detect'
    train_result_folder = os.path.join(detect_folder, 'train') 

    if TRAIN['Logo'] or not os.path.exists(train_result_folder):
        model = YOLO("yolov8n.pt")  
        results = model.train(data="data.yaml", epochs=logo_yolo_params['epochs'], batch=logo_yolo_params['batch'], imgsz=logo_yolo_params['img_size'])

    weights_path = os.path.join(train_result_folder, 'weights/best.pt')
    if INFERENCE['Logo']:
        model = YOLO(weights_path)
        img_folder = os.path.join('..', img_folder)
        for img_name in os.listdir(img_folder):
            results = model(os.path.join(img_folder, img_name), save=True, save_txt=True, save_conf=True, project=detect_folder, name=f'predict_{sample_folder_name}', exist_ok=True)

    os.chdir('..')


# def logo_detection_yolov8(logo_yolo_folder, logo_yolo_params, img_folder, TRAIN, INFERENCE):

#     detect_folder = os.path.join(logo_yolo_folder, 'runs/detect')
#     train_result_folder = os.path.join(detect_folder, 'train') 

#     if TRAIN['Logo'] or not os.path.exists(train_result_folder):
#         model = YOLO("yolov8n.pt")  
#         results = model.train(data="data.yaml", epochs=logo_yolo_params['epochs'], batch=logo_yolo_params['batch'], imgsz=logo_yolo_params['img_size'])

#     weights_path = os.path.join(train_result_folder, 'weights/best.pt')
#     if INFERENCE['Logo']:
#         model = YOLO(weights_path)
#         for img_name in os.listdir(img_folder):

#             results = model(os.path.join(img_folder, img_name), save=True, save_txt=True)

#             # 匹配以 predict 開頭的所有檔案或資料夾
#             logo_detect_result_path = glob(os.path.join(detect_folder, 'predict*'))[-1]   # 最新預測結果路徑: runs/segment/predict{last}
#             logo_labels_folder = os.path.join(logo_detect_result_path, 'labels')
#             os.makedirs(logo_labels_folder, exist_ok=True)

#             label_file_name = img_name.split('.')[0] + ".txt"
#             label_path = os.path.join(logo_labels_folder, label_file_name)

#             with open(label_path, "w") as f:
#                 for result in results:
#                     for box in result.boxes:
#                         x1, y1, x2, y2 = box.xyxy[0]  # 取得邊界框座標 (x1, y1, x2, y2)
#                         confidence = box.conf[0].item()  # 置信度
#                         class_id = int(box.cls[0].item())  # 類別 ID
                        
#                         # 計算中心座標與寬高
#                         x_center = (x1 + x2) / 2
#                         y_center = (y1 + y2) / 2
#                         width = x2 - x1
#                         height = y2 - y1
                        
#                         # 格式化輸出
#                         line = f"{class_id}, {x_center:.4f}, {y_center:.4f}, {width:.4f}, {height:.4f}, {confidence:.4f}\n"
#                         f.write(line)