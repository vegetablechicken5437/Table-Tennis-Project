import os
from glob import glob
from ultralytics import YOLO

def ball_detection_yolov5(ball_yolo_folder, ball_yolo_params, input_folder, all_sample_folder_name, sample_folder_name, TRAIN, INFERENCE):
    train_py = os.path.join(ball_yolo_folder, 'train.py')
    detect_py = os.path.join(ball_yolo_folder, 'detect.py')
    train_result_folder = os.path.join(ball_yolo_folder, 'runs/train')

    if TRAIN['Ball'] or not os.path.exists(train_result_folder):
        os.system(f'python {train_py} --img {ball_yolo_params["img_size"]} --batch {ball_yolo_params["batch"]} --epochs {ball_yolo_params["epochs"]} --data {ball_yolo_folder}/data.yaml --weights {ball_yolo_folder}/yolov5s.pt')
    
    weights_path = os.path.join(train_result_folder, 'exp/weights/best.pt')
    if INFERENCE['Ball']:
        os.system(f'python {detect_py} --weights {weights_path} --img {ball_yolo_params['img_size']} --source {input_folder} --save-txt --save-conf --project {ball_yolo_folder}/runs/detect/{all_sample_folder_name} --name exp_{sample_folder_name} --exist-ok')
        # os.system(f'python {detect_py} --weights {weights_path} --img {ball_yolo_params['img_size']} --source {img_folder} --save-txt --save-conf --project {ball_yolo_folder}/runs/detect --name exp_{sample_folder_name} --exist-ok')

def logo_detection_yolov8(logo_yolo_folder, logo_yolo_params, input_folder, all_sample_folder_name, sample_folder_name, TRAIN, INFERENCE):

    detect_folder = os.path.join(logo_yolo_folder, 'runs/detect')
    train_result_folder = os.path.join(detect_folder, 'train') 

    if TRAIN['Logo'] or not os.path.exists(train_result_folder):
        model = YOLO("yolov8n.pt")  
        results = model.train(data="data.yaml", epochs=logo_yolo_params['epochs'], batch=logo_yolo_params['batch'], imgsz=logo_yolo_params['img_size'])

    weights_path = os.path.join(train_result_folder, 'weights/best.pt')
    if INFERENCE['Logo']:
        model = YOLO(weights_path)
        for img_name in os.listdir(input_folder):
            results = model(os.path.join(input_folder, img_name), save=True, save_txt=True, save_conf=True, project=f'{detect_folder}/predict/{all_sample_folder_name}', name=sample_folder_name, exist_ok=True)
