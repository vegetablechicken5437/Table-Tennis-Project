import bpy
# from scipy.ndimage import gaussian_filter1d

# # 清空場景中現有的物體 (可選)
# bpy.ops.object.select_all(action='SELECT')
# bpy.ops.object.delete(use_global=False)

# 讀取座標txt檔
file_path = r"C:\Users\jason\Desktop\TableTennisProject\OUTPUT\sample_report-1\3D_pts.txt"

# 讀取檔案並處理座標
coordinates = []
with open(file_path, 'r') as file:
    for line in file:
        coords = line.strip().split()
        x, y, z = map(float, coords)
        coordinates.append((x, y, z))

# 提取前四個點，計算中心點
first_four_points = coordinates[:4]
center_x = sum([coord[0] for coord in first_four_points]) / 4
center_y = sum([coord[1] for coord in first_four_points]) / 4
center_z = sum([coord[2] for coord in first_four_points]) / 4
center = (center_x, center_y, center_z)

# 計算前四個點的X和Y範圍，用來設置正方形平面的尺寸
min_x = min([coord[0] for coord in first_four_points])
max_x = max([coord[0] for coord in first_four_points])
min_y = min([coord[1] for coord in first_four_points])
max_y = max([coord[1] for coord in first_four_points])

# 動態計算正方形平面的寬度和長度
square_width = max_x - min_x
square_height = max_y - min_y

# 創建正方形平面，頂點基於前四個點的範圍
vertices = [
    (min_x, min_y, center_z),
    (max_x, min_y, center_z),
    (max_x, max_y, center_z),
    (min_x, max_y, center_z),
]

# 創建正方形平面
mesh_data = bpy.data.meshes.new("square_plane_mesh")
mesh_data.from_pydata(vertices, [], [(0, 1, 2, 3)])
mesh_data.update()

# 創建一個新的物體並將正方形平面網格附加到場景中
plane_object = bpy.data.objects.new("SquarePlane", mesh_data)
bpy.context.collection.objects.link(plane_object)

# 將剩餘的座標轉換為球體
coordinates = coordinates[4:]
# sigma = 2  # Smoothing parameter
# coordinates = gaussian_filter1d(coordinates, sigma=sigma, axis=0)

# 創建曲線資料
curve_data = bpy.data.curves.new('LineCurve', type='CURVE')
curve_data.dimensions = '3D'

# 創建曲線物體
curve_object = bpy.data.objects.new('LineObject', curve_data)
bpy.context.collection.objects.link(curve_object)

# 設置曲線的樣條
spline = curve_data.splines.new(type='POLY')
spline.points.add(len(coordinates) - 1)

# 將座標轉換為曲線點
for i, coord in enumerate(coordinates):
    x, y, z = coord
    spline.points[i].co = (x, y, z, 1.0)  # 最後一個值是權重

# 設置線條的粗細
curve_data.bevel_depth = 0.01  # 這裡設置線的粗細，可以根據需要調整

# 創建材質並設置為綠色
material = bpy.data.materials.new(name="GreenMaterial")
material.diffuse_color = (0.0, 1.0, 0.0, 1.0)  # 設置為綠色 (R, G, B, Alpha)
curve_object.data.materials.append(material)


# 創建材質並設置為綠色
material = bpy.data.materials.new(name="OrangeMaterial")
material.diffuse_color = (1.0, 0.5, 0.0, 1.0)  # 設置為橘色 (R, G, B, Alpha)

# for coord in coordinates:
#     x, y, z = coord
#     # 創建球體
#     bpy.ops.mesh.primitive_uv_sphere_add(radius=0.04, location=(x, y, z))
#     sphere_object = bpy.context.object
    
#     # 將材質應用到球體上
#     if len(sphere_object.data.materials):
#         sphere_object.data.materials[0] = material
#     else:
#         sphere_object.data.materials.append(material)

# 調整視角以方便查看所有生成的物體
for area in bpy.context.screen.areas:
    if area.type == 'VIEW_3D':
        for space in area.spaces:
            if space.type == 'VIEW_3D':
                space.shading.type = 'MATERIAL'
        break

print("正方形平面和所有球體已成功生成，球體基於所有座標點！")

