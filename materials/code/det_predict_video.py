# 引入必要的库和包
from ultralytics import YOLO
import cv2


# 加载模型
model = YOLO(model="yolov8n.pt")

# 查看模型
print(model.model)

#
video_path = "./girls.mp4"
# video_path = 1
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    status, frame = cap.read()
    if not status:
        break
    results = model.predict(source=frame)
    result = results[0]
    print(result)
    # 
    anno_frame = result.plot()
    cv2.imshow(winname="frame", mat=anno_frame)
    if cv2.waitKey(delay=100) == 27:
        break

cap.release()
cv2.destroyAllWindows()
