# 重新載入必要套件並繪製 3D 圖表
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# 範例資料：上輪速度、下輪速度與轉速
data = {
    '上輪速度': [5, 6.5, 8, 1, 1, 7],
    '下輪速度': [1, 1, 1, 6, 9, 1],
    '轉速': [1100, 1300, 1450, 750, 520, 1200]
}
df = pd.DataFrame(data)

# 特徵與目標變數
X = df[['上輪速度', '下輪速度']].values
y = df['轉速'].values

# 進行多項式回歸處理
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# 訓練回歸模型
model = LinearRegression()
model.fit(X_poly, y)

# 生成網格點並預測
x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 10)
y_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 10)
x_mesh, y_mesh = np.meshgrid(x_range, y_range)
z_mesh = model.predict(poly.transform(np.c_[x_mesh.ravel(), y_mesh.ravel()])).reshape(x_mesh.shape)

# 可視化結果
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(x_mesh, y_mesh, z_mesh, cmap='viridis', edgecolor='k')
ax.scatter(X[:, 0], X[:, 1], y, color='r', label='測量數據')
ax.set_xlabel('上輪速度')
ax.set_ylabel('下輪速度')
ax.set_zlabel('轉速 (rpm)')
ax.set_title('多項式回歸擬合結果')

# 顯示圖表
plt.legend()
plt.show()
