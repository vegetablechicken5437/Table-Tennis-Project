# 桌球軌跡追蹤與旋轉速度量測系統 (Table Tennis Trajectory Tracking and Spin Rate Estimation System)

此為一套可以重建3D軌跡以及計算旋轉速度的桌球訓練輔助系統。透過雙相機校正及同步、YOLO深度學習模型、立體視覺幾何推算與旋轉速度估算模型等核心技術，成功解決動態模糊下的特徵追蹤困難，並克服傳統旋轉速度量測方法的挑戰，實現3D軌跡重建及旋轉速度估算，為教練與選手提供精準的技術分析數據。

https://github.com/user-attachments/assets/37df1677-bd13-493f-acd3-e98be5b4bc2b

## 系統架構

### 1. 雙相機校正
- 透過拍攝棋盤格圖案進行雙相機標定。接著採用 **張氏標定法 (Zhang’s Camera Calibration Method)**，獲得以下參數：內部參數矩陣 **K**，外部參數 **R、T**，以及畸變係數，校正結果將用於後續立體視覺重建。
<img width="500" height="250" alt="image" src="https://github.com/user-attachments/assets/521114e2-e02a-4920-ae18-064a044d6a4c" />

### 2. YOLOv11 兩階段偵測
- **階段一：** 在完整畫面中偵測並追蹤桌球位置。
<img width="300" height="200" alt="image" src="https://github.com/user-attachments/assets/45594469-f1b4-4fd0-8774-2901ebd4ed89" />

- **階段二：** 在第一階段的結果中，進一步偵測球面標記，以分析旋轉方向與角度變化。
<img width="150" height="150" alt="image" src="https://github.com/user-attachments/assets/ca25e74d-c599-4dcc-a82d-4e81b0ab327f" />

- 兩階段皆使用 YOLOv11 訓練之自建資料集，適用於室內高速拍攝環境。

### 3. 立體視覺 3D 座標重建
- 根據雙相機校正所得投影矩陣進行三維重建，將左右畫面中對應的桌球與球面標記匹配後，透過三角測量求出其 3D 座標。
<img width="400" height="280" alt="image" src="https://github.com/user-attachments/assets/b9106e49-c9eb-4765-b115-c76d69857a01" />
<img width="400" height="320" alt="image" src="https://github.com/user-attachments/assets/fc9ab699-38a5-4a8a-aa13-f40afd67c926" />

### 4. 旋轉速度計算
- 根據連續幀中標記點的角度變化，計算旋轉位移 Δθ，結合時間差 Δt，估算旋轉角速度：
<img width="500" height="360" alt="image" src="https://github.com/user-attachments/assets/2253516b-6147-4b3c-a867-e79bee612f56" />



