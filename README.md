# 🏓 桌球旋轉速度偵測系統 (Table Tennis Spin Tracking System)

本專案旨在開發一套以 **雙相機與 YOLOv11 影像辨識技術** 為核心的 **桌球旋轉速度偵測系統**。  
透過雙相機立體視覺、攝影機標定、深度學習物件偵測與 3D 重建技術，實現桌球在空間中的 **旋轉軸與角速度估算**。

---

## 📸 系統概述

本系統採用 **兩台同步攝影機** 拍攝桌球運動影像，並結合 **YOLOv11 兩階段偵測** 與 **3D 重建演算法**，  
以高精度重建桌球的三維位置與旋轉資訊。

<p align="center">
  <img src="docs/system_overview.png" width="700">
</p>

---

## 🧩 系統功能特色

### 1️⃣ 雙相機校正 (Dual-Camera Calibration)
- 透過拍攝棋盤格 (Checkerboard) 圖案進行雙相機標定。  
- 採用 **張氏標定法 (Zhang’s Camera Calibration Method)**，獲得以下參數：
  - 內部參數矩陣 **K**
  - 外部參數 **R、T**
  - 畸變係數 **Distortion coefficients**
- 標定結果將用於後續立體視覺重建。

---

### 2️⃣ YOLOv11 兩階段偵測 (Two-Stage Detection)
- **階段一：** 偵測並追蹤 **桌球位置**。  
- **階段二：** 偵測 **球面標記 (如叉叉或十字圖案)** 以分析旋轉方向與角度變化。  

> 兩階段皆使用 YOLOv11 訓練之自建資料集，適用於室內高速拍攝環境。

---

### 3️⃣ 立體視覺 3D 座標重建 (Stereo Vision 3D Reconstruction)
- 根據雙相機標定所得投影矩陣進行三維重建。  
- 將左右畫面中對應的桌球點匹配後，透過三角測量 (Triangulation) 求出其 3D 座標。

公式如下：
\[
X = \text{triangulate}(K_1, R_1, T_1, K_2, R_2, T_2, x_1, x_2)
\]

---

### 4️⃣ 旋轉速度計算 (Spin Speed Estimation)
- 根據連續幀中標記點的角度變化，計算旋轉位移 Δθ。  
- 結合時間差 Δt，估算旋轉角速度：
\[
\omega = \frac{\Delta \theta}{\Delta t}
\]
- 可視化輸出包括：
  - 桌球旋轉軸方向  
  - 每幀角速度曲線  
  - 三維旋轉軌跡重建圖  

---

## ⚙️ 系統需求

| 套件 | 版本 |
|------|------|
| Python | 3.8+ |
| OpenCV | ≥ 4.8 |
| PyTorch | ≥ 2.2 |
| NumPy / Matplotlib | 最新版 |
| YOLOv11 | Ultralytics 版本 |

---

## 🚀 執行步驟

```bash
# 1️⃣ 複製專案
git clone https://github.com/yourname/table-tennis-spin-tracking.git
cd table-tennis-spin-tracking

# 2️⃣ 執行雙相機校正
python calibrate_dual_camera.py --input ./calib_images/

# 3️⃣ 使用 YOLOv11 進行桌球與標記偵測
python detect_ball_and_mark.py --input ./videos/ --weights ./weights/yolov11.pt

# 4️⃣ 進行 3D 重建與旋轉速度計算
python compute_spin_speed.py
