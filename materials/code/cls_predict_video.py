"""
cls_predict_video - 图像分类模型预测

Author: liuzhenzhen
Date: 2024/1/31
"""

from ultralytics import YOLO
import cv2

# 加载模型
model = YOLO(model="yolov8n-cls.pt")

# 从视频文件中预测
video_path = "./girls.mp4"
cap = cv2.VideoCapture(video_path)

# 从摄像头中预测
# webcam = 0
# cap = cv2.VideoCapture(webcam)


while cap.isOpened():
    status, frame = cap.read()
    if not status:
        break
    # 
    results = model.predict(source=frame)
    result = results[0]

    anno_frame = result.plot()
    cv2.imshow(winname="frame", mat=anno_frame)

    # 按键 ESC 退出
    if cv2.waitKey(delay=100) == 27:
        break

cap.release()
cv2.destroyAllWindows()
