import cv2
import time
import numpy as np
from ultralytics import YOLO

model = YOLO('yolo26m.pt')
cap = cv2.VideoCapture("dashcam_sample.mp4")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter('epic_spatial_radar.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width * 2, frame_height))

KNOWN_WIDTH = 1.8 
FOCAL_LENGTH = 800
MAX_RADAR_DIST = 100 

def estimate_distance(pixel_width):
    if pixel_width == 0: 
        return 0
    return (KNOWN_WIDTH * FOCAL_LENGTH) / pixel_width

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Spatial processing complete!")
        break

    radar_bg = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

    horizon_y = int(frame_height * 0.4)
    cv2.line(radar_bg, (0, horizon_y), (frame_width, horizon_y), (255, 100, 0), 2)
    center_x = int(frame_width / 2)

    for i in range(-5, 6):
        start_x = int(center_x + (i * frame_width / 10))
        cv2.line(radar_bg, (center_x, horizon_y), (start_x, frame_height), (255, 50, 0), 1)
    for j in range(1, 6):
        y = horizon_y + int((frame_height - horizon_y) * (j / 5.0)**2)
        cv2.line(radar_bg, (0, y), (frame_width, y), (255, 50, 0), 1)

    results = model(frame, classes=[1, 2, 3, 5, 7])

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            pixel_width = x2 - x1
            
            distance = estimate_distance(pixel_width)
            obj_center_x = int((x1 + x2) / 2)

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            safe_dist = min(distance, MAX_RADAR_DIST)
            depth_ratio = 1 - (safe_dist / MAX_RADAR_DIST)
            
            radar_y = horizon_y + int((frame_height - horizon_y) * (depth_ratio**2))
            
            box_size = int(30 * depth_ratio) + 5
            
            cv2.rectangle(radar_bg, (obj_center_x - box_size, radar_y - box_size), 
                          (obj_center_x + box_size, radar_y), (0, 255, 0), 2)
            cv2.putText(radar_bg, f"{distance:.1f}m", (obj_center_x - box_size, radar_y - box_size - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    combined_frame = np.hstack((frame, radar_bg))

    out.write(combined_frame)
    print("Processing spatial frame...")

cap.release()
out.release()