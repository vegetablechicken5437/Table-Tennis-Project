import os
from glob import glob
from ultralytics import YOLO

def ball_detection_yolov5(ball_yolo_folder, ball_yolo_params, input_folder, all_sample_folder_name, sample_folder_name, TRAIN, INFERENCE):
    train_py = os.path.join(ball_yolo_folder, 'train.py')
    detect_py = os.path.join(ball_yolo_folder, 'detect.py')
    train_result_folder = os.path.join(ball_yolo_folder, 'runs/train')

    # 執行訓練
    if TRAIN['Ball'] or not os.path.exists(train_result_folder):
        os.system(f'python {train_py} --img {ball_yolo_params["img_size"]} --batch {ball_yolo_params["batch"]} --epochs {ball_yolo_params["epochs"]} --data {ball_yolo_folder}/data.yaml --weights {ball_yolo_folder}/yolov5s.pt')

    # 找到最後一個 exp 資料夾
    def get_last_exp_folder(path):
        exps = [d for d in os.listdir(path) if d.startswith('exp') and os.path.isdir(os.path.join(path, d))]
        exps = sorted(exps, key=lambda x: int(x[3:]) if x[3:].isdigit() else float('-inf'))
        # print(exps)
        return os.path.join(path, exps[-1]) if exps else None

    latest_exp_folder = get_last_exp_folder(train_result_folder)
    weights_path = os.path.join(latest_exp_folder, 'weights', 'best.pt') if latest_exp_folder else None

    # 推論階段
    if INFERENCE['Ball'] and weights_path:
        print(f'[INFO] 使用的模型權重為: {weights_path}')  # 印出使用的 best.pt
        os.system(f'python {detect_py} --weights {weights_path} --img {ball_yolo_params["img_size"]} --source {input_folder} --save-txt --save-conf --project {ball_yolo_folder}/runs/detect/{all_sample_folder_name} --name exp_{sample_folder_name} --exist-ok')

def logo_detection_yolov8(logo_yolo_folder, logo_yolo_params, input_folder, all_sample_folder_name, sample_folder_name, TRAIN, INFERENCE):

    if TRAIN['Logo']:
        os.chdir(logo_yolo_folder)
        model = YOLO("yolov8n-seg.pt")  
        results = model.train(data="data.yaml", epochs=logo_yolo_params['epochs'], batch=logo_yolo_params['batch'], imgsz=logo_yolo_params['img_size'])
        os.chdir('..')

    detect_folder = f'{logo_yolo_folder}/runs/segment'  # detect 裡面的 train、train2、train3...

    # 找出最後一個 train 資料夾（train, train2, ...）
    def get_last_train_folder(path):
        trains = [d for d in os.listdir(path) if d.startswith('train') and os.path.isdir(os.path.join(path, d))]
        trains = sorted(trains, key=lambda x: int(x[3:]) if x[3:].isdigit() else float('-inf'))
        print(trains)
        return os.path.join(path, trains[-1]) if trains else None

    latest_train_folder = get_last_train_folder(detect_folder)
    weights_path = os.path.join(latest_train_folder, 'weights', 'best.pt') if latest_train_folder else None

    if INFERENCE['Logo'] and weights_path:
        print(f'[INFO] 使用的模型權重為: {weights_path}')
        model = YOLO(weights_path)
        for img_name in os.listdir(input_folder):
            img_path = os.path.join(input_folder, img_name)
            results = model(img_path, save=True, save_txt=True, save_conf=True,
                            project=os.path.join(detect_folder, 'predict', all_sample_folder_name),
                            name=sample_folder_name, exist_ok=True)
