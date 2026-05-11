import cv2
import time
from ultralytics import YOLO

model = YOLO('yolo26m.pt')
cap = cv2.VideoCapture("traffic_sample.mp4")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))


out = cv2.VideoWriter('output_radar.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

KNOWN_WIDTH = 1.8 
FOCAL_LENGTH = 800

previous_frame_cars = {} 
car_id_counter = 0
fps_time = time.time()

def estimate_distance(pixel_width):
    if pixel_width == 0: 
        return 0
    
    distance = (KNOWN_WIDTH * FOCAL_LENGTH) / pixel_width
    return distance

def calculate_speed(dist1, dist2, time_elapsed):
    if time_elapsed == 0:
        return 0
    # Difference in distance (in meters)
    distance_change = abs(dist1 - dist2)
    # Speed in meters per second
    speed_mps = distance_change / time_elapsed
    # Convert to km/h
    speed_kmh = speed_mps * 3.6
    return speed_kmh

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Video processing complete!")
        break

    start_time = time.time()
    
    results = model(frame, classes=[1, 2, 3, 5, 7])

    # Calculate time passed since the last frame
    current_time = time.time()
    time_elapsed = current_time - fps_time
    fps_time = current_time

    current_frame_cars = {}
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            pixel_width = x2 - x1
            
            # 1. Calculate Distance
            distance = estimate_distance(pixel_width)
            
            # 2. Calculate Center Point
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            # 3. Tracking Logic: Find if we saw this car in the previous frame
            matched_id = None
            min_dist = float('inf')
            
            for car_id, (prev_x, prev_y, prev_dist) in previous_frame_cars.items():
                center_dist = ((center_x - prev_x)**2 + (center_y - prev_y)**2)**0.5
                
                if center_dist < 50 and center_dist < min_dist:
                    min_dist = center_dist
                    matched_id = car_id

            speed = 0
            if matched_id is not None:
                prev_dist = previous_frame_cars[matched_id][2]
                speed = calculate_speed(prev_dist, distance, time_elapsed)
                current_frame_cars[matched_id] = (center_x, center_y, distance)
            else:
                matched_id = car_id_counter
                car_id_counter += 1
                current_frame_cars[matched_id] = (center_x, center_y, distance)

            
            SPEED_LIMIT = 70
            
            if speed > SPEED_LIMIT:
                hud_color = (0, 0, 255)  # BGR format: Red
            else:
                hud_color = (0, 255, 0)  # BGR format: Green
                
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), hud_color, 2)
            
            text = f"ID:{matched_id} | {distance:.1f}m | {speed:.1f}km/h"
            
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            cv2.rectangle(frame, (int(x1), int(y1) - text_height - 10), (int(x1) + text_width, int(y1)), hud_color, -1)
            
            cv2.putText(frame, text, (int(x1), int(y1) - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


    previous_frame_cars = current_frame_cars


    out.write(frame)
    
    print("Processing frame...")

cap.release()
out.release()
cv2.destroyAllWindows()