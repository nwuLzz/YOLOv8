# 引入必要的库和包
from ultralytics import YOLO
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if __name__ == "__main__":
    # 构建模型
    model = YOLO(model="yolov8n-cls.yaml")

    # 训练模型
    model.train(data='minst160', epochs=10, imgsz=64)
