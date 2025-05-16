import os
from glob import glob
from ultralytics import YOLO

# def ball_detection_yolov5(ball_yolo_folder, ball_yolo_params, input_folder, 
#                           all_sample_folder_name, sample_folder_name, TRAIN, INFERENCE):
    
#     train_py = os.path.join(ball_yolo_folder, 'train.py')
#     detect_py = os.path.join(ball_yolo_folder, 'detect.py')
#     train_result_folder = os.path.join(ball_yolo_folder, 'runs/train')

#     # 執行訓練
#     if TRAIN['Ball'] or not os.path.exists(train_result_folder):
#         os.system(f'python {train_py} --img {ball_yolo_params["img_size"]} --batch {ball_yolo_params["batch"]} --epochs {ball_yolo_params["epochs"]} --data {ball_yolo_folder}/data.yaml --weights {ball_yolo_folder}/yolov5s.pt')

#     # 找到最後一個 exp 資料夾
#     def get_last_exp_folder(path):
#         exps = [d for d in os.listdir(path) if d.startswith('exp') and os.path.isdir(os.path.join(path, d))]
#         exps = sorted(exps, key=lambda x: int(x[3:]) if x[3:].isdigit() else float('-inf'))
#         return os.path.join(path, exps[-1]) if exps else None

#     latest_exp_folder = get_last_exp_folder(train_result_folder)
#     weights_path = os.path.join(latest_exp_folder, 'weights', 'best.pt') if latest_exp_folder else None

#     # 推論階段
#     if INFERENCE['Ball'] and weights_path:
#         print(f'[INFO] 使用的模型權重為: {weights_path}')  # 印出使用的 best.pt
#         os.system(f'python {detect_py} --weights {weights_path} --img {ball_yolo_params["img_size"]} --source {input_folder} --save-txt --save-conf --project {ball_yolo_folder}/runs/detect/{all_sample_folder_name} --name exp_{sample_folder_name} --exist-ok')


# def ball_detection_yolov11(ball_yolo_folder, ball_yolo_params, input_folder, 
#                            all_sample_folder_name, sample_folder_name, TRAIN, INFERENCE):
    
#     data_yaml_path = os.path.join(ball_yolo_folder, 'data.yaml')
#     weights_init = os.path.join(ball_yolo_folder, 'yolo11n.pt')
#     model_output_folder = os.path.join(ball_yolo_folder, 'runs/detect', all_sample_folder_name, f'exp_{sample_folder_name}')
    
#     # ===========================
#     # 1. 訓練階段
#     # ===========================
#     if TRAIN['Ball']:
#         print(f"[INFO] 開始訓練 YOLOv11 模型")
#         model = YOLO(weights_init)
#         model.train(
#             data=data_yaml_path,
#             epochs=ball_yolo_params["epochs"],
#             imgsz=ball_yolo_params["img_size"],
#             batch=ball_yolo_params["batch"],
#             project=os.path.join(ball_yolo_folder, 'runs/train'),
#             name='exp',
#             exist_ok=True
#         )

#     # ===========================
#     # 2. 找到最新 best.pt
#     # ===========================
#     def get_last_exp_folder(path):
#         exps = [d for d in os.listdir(path) if d.startswith('exp') and os.path.isdir(os.path.join(path, d))]
#         exps = sorted(exps, key=lambda x: int(x[3:]) if x[3:].isdigit() else float('-inf'))
#         return os.path.join(path, exps[-1]) if exps else None

#     latest_exp_folder = get_last_exp_folder(os.path.join(ball_yolo_folder, 'runs/train'))
#     weights_path = os.path.join(latest_exp_folder, 'weights', 'best.pt') if latest_exp_folder else None

#     # ===========================
#     # 3. 推論階段
#     # ===========================
#     if INFERENCE['Ball'] and weights_path:
#         print(f'[INFO] 使用的模型權重為: {weights_path}')
#         model = YOLO(weights_path)

#         results = model.predict(
#             source=input_folder,
#             imgsz=ball_yolo_params["img_size"],
#             save=True,
#             save_txt=True,
#             save_conf=True,
#             project=os.path.join(ball_yolo_folder, 'runs/detect', all_sample_folder_name),
#             name=f'exp_{sample_folder_name}',
#             exist_ok=True
#         )

#         print(f"[INFO] 推論完成，label 檔儲存於：{model_output_folder}")

# def mark_detection_yolov8(logo_yolo_folder, logo_yolo_params, input_folder, 
#                           all_sample_folder_name, sample_folder_name, TRAIN, INFERENCE):

#     data_yaml_path = os.path.join(logo_yolo_folder, 'data.yaml')
#     weights_init = os.path.join(logo_yolo_folder, 'yolo11n-seg.pt')

#     if TRAIN['Logo']:
#         print(f"[INFO] 開始訓練 YOLOv11 模型")
#         model = YOLO(weights_init)  
#         model.train(
#             data=data_yaml_path,
#             epochs=logo_yolo_params["epochs"], 
#             imgsz=logo_yolo_params["img_size"],
#             batch=logo_yolo_params["batch"]
#         )

#     detect_folder = f'{logo_yolo_folder}/runs/segment'  # detect 裡面的 train、train2、train3...

#     # 找出最後一個 train 資料夾（train, train2, ...）
#     def get_last_train_folder(path):
#         trains = [d for d in os.listdir(path) if d.startswith('train') and os.path.isdir(os.path.join(path, d))]
#         trains = sorted(trains, key=lambda x: int(x[3:]) if x[3:].isdigit() else float('-inf'))
#         return os.path.join(path, trains[-1]) if trains else None

#     latest_train_folder = get_last_train_folder(detect_folder)
#     weights_path = os.path.join(latest_train_folder, 'weights', 'best.pt') if latest_train_folder else None

#     if INFERENCE['Logo'] and weights_path:
#         print(f'[INFO] 使用的模型權重為: {weights_path}')
#         model = YOLO(weights_path)
#         for img_name in os.listdir(input_folder):
#             img_path = os.path.join(input_folder, img_name)
#             results = model(img_path, save=True, save_txt=True, save_conf=True,
#                             project=os.path.join(detect_folder, 'predict', all_sample_folder_name),
#                             name=sample_folder_name, exist_ok=True)

def ball_detection_yolov11_training(ball_yolo_folder, ball_yolo_params):
    
    data_yaml_path = os.path.join(ball_yolo_folder, 'data.yaml')
    weights_init = os.path.join(ball_yolo_folder, 'yolo11n.pt')
    model_output_folder = os.path.join(ball_yolo_folder, 'runs/detect')
    
    print(f"[INFO] 開始訓練 YOLOv11 模型")
    model = YOLO(weights_init)
    model.train(
        data=data_yaml_path,
        epochs=ball_yolo_params["epochs"],
        imgsz=ball_yolo_params["img_size"],
        batch=ball_yolo_params["batch"],
        project=model_output_folder
    )

def ball_detection_yolov11_inferencing(ball_yolo_folder, ball_yolo_params, 
                                       input_folder, all_sample_folder_name, sample_folder_name):

    model_output_folder = os.path.join(ball_yolo_folder, 'runs/detect')
    project_folder_path = os.path.join(model_output_folder, 'predict', all_sample_folder_name)

    # 找出最後一個 train 資料夾（train, train2, ...）
    def get_last_train_folder(path):
        trains = [d for d in os.listdir(path) if d.startswith('train') and os.path.isdir(os.path.join(path, d))]
        trains = sorted(trains, key=lambda x: int(x[3:]) if x[3:].isdigit() else float('-inf'))
        return os.path.join(path, trains[-1]) if trains else None

    latest_train_folder = get_last_train_folder(model_output_folder)
    weights_path = os.path.join(latest_train_folder, 'weights', 'best.pt') if latest_train_folder else None

    if weights_path:
        print(f'[INFO] 使用的模型權重為: {weights_path}')
        model = YOLO(weights_path)
        model.predict(
            source=input_folder,
            imgsz=ball_yolo_params["img_size"],
            save=True,
            save_txt=True,
            save_conf=True,
            project=project_folder_path,
            name=sample_folder_name, 
            exist_ok=True
        )
        print(f"[INFO] 推論完成，label 檔儲存於：{project_folder_path}")
    else:
        print(f"[INFO] 尚未訓練模型")

def mark_detection_yolov11_training(mark_yolo_folder, mark_yolo_params):

    data_yaml_path = os.path.join(mark_yolo_folder, 'data.yaml')
    weights_init = os.path.join(mark_yolo_folder, 'yolo11n-seg.pt')
    model_output_folder = os.path.join(mark_yolo_folder, 'runs/segment')

    print(f"[INFO] 開始訓練 YOLOv11 模型")
    model = YOLO(weights_init)  
    model.train(
        data=data_yaml_path,
        epochs=mark_yolo_params["epochs"], 
        imgsz=mark_yolo_params["img_size"],
        batch=mark_yolo_params["batch"],
        project=model_output_folder
    )

def mark_detection_yolov11_inferencing(mark_yolo_folder, mark_yolo_params, 
                                       input_folder, all_sample_folder_name, sample_folder_name):

    model_output_folder = os.path.join(mark_yolo_folder, 'runs/segment')    # segment 裡面的 train、train2、train3...
    project_folder_path = os.path.join(model_output_folder, 'predict', all_sample_folder_name)

    # 找出最後一個 train 資料夾（train, train2, ...）
    def get_last_train_folder(path):
        trains = [d for d in os.listdir(path) if d.startswith('train') and os.path.isdir(os.path.join(path, d))]
        trains = sorted(trains, key=lambda x: int(x[3:]) if x[3:].isdigit() else float('-inf'))
        return os.path.join(path, trains[-1]) if trains else None

    latest_train_folder = get_last_train_folder(model_output_folder)
    weights_path = os.path.join(latest_train_folder, 'weights', 'best.pt') if latest_train_folder else None

    if weights_path:
        print(f'[INFO] 使用的模型權重為: {weights_path}')
        model = YOLO(weights_path)
        model.predict(
            source=input_folder, 
            imgsz=mark_yolo_params["img_size"],
            save=True, 
            save_txt=True, 
            save_conf=True,
            project=project_folder_path,
            name=sample_folder_name
        )
        print(f"[INFO] 推論完成，label 檔儲存於：{project_folder_path}")
    else:
        print(f"[INFO] 尚未訓練模型")

if __name__ == "__main__":
    pass