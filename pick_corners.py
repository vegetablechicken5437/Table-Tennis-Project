import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk

class CornerPicker:
    def __init__(self, points, output_folder):
        self.points = points
        self.img_left = None
        self.img_right = None
        self.img_left_resized = None
        self.img_right_resized = None
        self.wL = None
        self.scaleL = None
        self.scaleR = None
        self.root = None
        self.panel_L = None
        self.panel_R = None
        self.img_display_L = None
        self.img_display_R = None
        self.left_corners = None
        self.right_corners = None
        self.output_folder = output_folder

    def resize_image(self, image, max_width, max_height):
        h, w = image.shape[:2]
        scale = min(max_width / w, max_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(image, (new_w, new_h)), new_w, new_h, scale

    def draw_grid(self, img, scale, step=50):
        h, w = img.shape[:2]
        for x in range(0, w, step):
            cv2.line(img, (x, 0), (x, h), (100, 100, 100), 1)
            cv2.putText(img, f"{int(x/scale)}", (x+2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        for y in range(0, h, step):
            cv2.line(img, (0, y), (w, y), (100, 100, 100), 1)
            cv2.putText(img, f"{int(y/scale)}", (2, y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    def draw_points(self, img, points, scale, offset_x=0):
        for i, point in enumerate(points):
            color = (0, 0, 255) if i == 0 else (0, 255, 255)
            adjusted_point = (int(point[0] * scale - offset_x * scale), int(point[1] * scale))
            cv2.circle(img, adjusted_point, 5, color, -1)
            cv2.putText(img, f"P{i+1} ({int(point[0] - offset_x)}, {int(point[1])})", 
                        (adjusted_point[0] + 5, adjusted_point[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def draw_points_final(self, img, points, offset_x=0):
        for i, point in enumerate(points):
            color = (0, 0, 255) if i == 0 else (0, 255, 255)
            point = int(point[0] - offset_x), int(point[1])
            cv2.circle(img, point, 5, color, -1)
            cv2.putText(img, f"P{i+1} ({int(point[0])}, {int(point[1])})", 
                        (point[0] + 5, point[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if x < self.wL and len([p for p in self.points if p[0] < self.wL/self.scaleL]) < 4:
                self.points.append((x / self.scaleL, y / self.scaleL))
            elif x >= self.wL and len([p for p in self.points if p[0] >= self.wL/self.scaleL]) < 4:
                self.points.append((x / self.scaleR, y / self.scaleR))
            self.update_images()

    def delete_last_point(self):
        if self.points:
            self.points.pop()
            self.update_images()

    def save_and_exit(self):
        os.makedirs(self.output_folder, exist_ok=True)
        self.left_corners = np.array([[int(p[0]), int(p[1])] for p in self.points if p[0] < self.wL / self.scaleL])
        self.right_corners = np.array([[int(p[0] - self.wL / self.scaleL), int(p[1])] for p in self.points if p[0] >= self.wL / self.scaleL])
        np.savetxt(os.path.join(self.output_folder, "left_corners.txt"), self.left_corners)
        np.savetxt(os.path.join(self.output_folder, "right_corners.txt"), self.right_corners)
        print(f'左圖球桌角落: {self.left_corners.tolist()}')
        print(f'右圖球桌角落: {self.right_corners.tolist()}')

        img_L_copy = self.img_left.copy()
        img_R_copy = self.img_right.copy()
        self.draw_grid(img_L_copy, 1)
        self.draw_grid(img_R_copy, 1)
        self.draw_points_final(img_L_copy, [p for p in self.points if p[0] < self.wL / self.scaleL], offset_x=0)
        self.draw_points_final(img_R_copy, [p for p in self.points if p[0] >= self.wL / self.scaleL], offset_x=self.wL/self.scaleL)
        cv2.imwrite(os.path.join(self.output_folder, "marked_left.jpg"), img_L_copy)
        cv2.imwrite(os.path.join(self.output_folder, "marked_right.jpg"), img_R_copy)
        self.root.quit()
        self.root.destroy()

    def update_images(self):
        img_L_copy = self.img_left_resized.copy()
        img_R_copy = self.img_right_resized.copy()
        self.draw_points(img_L_copy, [p for p in self.points if p[0] < self.wL / self.scaleL], self.scaleL)
        self.draw_points(img_R_copy, [p for p in self.points if p[0] >= self.wL / self.scaleL], self.scaleR, offset_x=self.wL/self.scaleL)
        self.img_display_L = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(img_L_copy, cv2.COLOR_BGR2RGB)))
        self.img_display_R = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(img_R_copy, cv2.COLOR_BGR2RGB)))
        self.panel_L.config(image=self.img_display_L)
        self.panel_R.config(image=self.img_display_R)

    def pick_corners(self, enhanced_img_folder):
        self.root = tk.Tk()
        self.root.withdraw()
        os.makedirs(self.output_folder, exist_ok=True)
        files = sorted(os.listdir(enhanced_img_folder))
        left_img_name = next((f for f in files if 'L' in f), None)
        right_img_name = next((f for f in files if 'R' in f), None)
        if not left_img_name or not right_img_name:
            print("找不到合適的 L 和 R 圖片！")
            return
        self.img_left = cv2.imread(os.path.join(enhanced_img_folder, left_img_name))
        self.img_right = cv2.imread(os.path.join(enhanced_img_folder, right_img_name))
        self.img_left_resized, self.wL, _, self.scaleL = self.resize_image(self.img_left, 600, 800)
        self.img_right_resized, _, _, self.scaleR = self.resize_image(self.img_right, 600, 800)
        self.draw_grid(self.img_left_resized, self.scaleL)
        self.draw_grid(self.img_right_resized, self.scaleR)
        self.root.deiconify()
        self.img_display_L = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(self.img_left_resized, cv2.COLOR_BGR2RGB)))
        self.img_display_R = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(self.img_right_resized, cv2.COLOR_BGR2RGB)))
        self.root.title("標記球桌角點")
        self.panel_L = Label(self.root, image=self.img_display_L)
        self.panel_L.grid(row=0, column=0, padx=10, pady=10)
        self.panel_L.bind("<Button-1>", lambda event: self.mouse_callback(cv2.EVENT_LBUTTONDOWN, event.x, event.y, None, None))
        self.panel_R = Label(self.root, image=self.img_display_R)
        self.panel_R.grid(row=0, column=1, padx=10, pady=10)
        self.panel_R.bind("<Button-1>", lambda event: self.mouse_callback(cv2.EVENT_LBUTTONDOWN, event.x + self.wL, event.y, None, None))
        Button(self.root, text="刪除前一個點", command=self.delete_last_point).grid(row=1, column=0, pady=10)
        Button(self.root, text="完成選取並儲存", command=self.save_and_exit).grid(row=1, column=1, pady=10)
        self.root.mainloop()

if __name__ == "__main__":
    picker = CornerPicker([], ".")
    picker.pick_corners('ProcessedImages/0412/20250412_152611/enhanced_LR')
