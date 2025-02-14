from ultralytics import YOLO
import os
model = YOLO('yolov5s.pt')
video_path = 'assets/vid3.mp4'


video_path = 'C:/Users/REDA/Desktop/SUMMER/SEP 2024/Football Analysis/assets/vid3.mp4'
if os.path.exists(video_path):
    print(f"File {video_path} found!")
    results = model.predict(video_path, save=True)
else:
    print(f"File {video_path} does not exist")


print(results[0])

for box in results[0].boxes:
    print(box)