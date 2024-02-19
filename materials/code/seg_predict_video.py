"""
seg_predict_video - 实例分割模型预测

Author: liuzhenzhen
Date: 2024/1/31
"""

from ultralytics import YOLO
import cv2

# 加载模型
model = YOLO(model="yolov8n-seg.pt")

# 查看模型
# print(model.model)

# 从视频文件中预测
video_path = "./girls.mp4"
# # 从摄像头中预测
# video_path = 1
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    status, frame = cap.read()

    if not status:
        break

    # results = model.predict(source=frame)       # 实例分割
    results = model.track(source=frame)       # 物体追踪
    result = results[0]
    # # 打印结果
    # print(result)
    # 结果画到图像上
    anno_frame = result.plot()
    cv2.imshow(winname="frame", mat=anno_frame)     # 显示图像

    # 按Esc退出
    if cv2.waitKey(delay=10) == 27:
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
