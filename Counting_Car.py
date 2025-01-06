import cv2
import cvzone
from ultralytics import YOLO
import math
from sort import *
import numpy as np

# Video
# cap = cv2.VideoCapture("./Videos/test3.mp4")
# cap = cv2.VideoCapture("./Videos/cars.mp4")
cap = cv2.VideoCapture("./Videos/v4.mp4")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)

# YOLO Model
model = YOLO("./YOLO_Weights/yolov8l.pt")

# Class Names (COCO)
coco_classes = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
    "pottedplant", "bed", "diningtable", "toilet", "TV monitor", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# SORT Tracker
tracker = Sort(max_age=20, min_hits=1, iou_threshold=0.3)

# Tracking Line Position
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
tracking_line_y = 520  # Adjust this value to move the tracking line up or down
tracking_line = [0, tracking_line_y, frame_width, tracking_line_y]

# Vehicle Count
count = []
vehicle_directions = {}

# Main Loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame, stream=True)
    detections = np.empty((0, 5))

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            confidence = round(float(box.conf[0]), 2)
            cls = int(box.cls[0])
            current_class = coco_classes[cls]

            # Filter for vehicles
            if current_class in ["car", "truck", "bus", "motorbike"] and confidence > 0.2:
                detections = np.vstack((detections, [x1, y1, x2, y2, confidence]))

    # Update tracker
    tracked_results = tracker.update(detections)

    # Draw tracking line
    cv2.line(frame, (tracking_line[0], tracking_line[1]), (tracking_line[2], tracking_line[3]), (255, 0, 255), 4)

    for result in tracked_results:
        x1, y1, x2, y2, obj_id = map(int, result)
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2

        # Draw bounding box and ID
        cvzone.cornerRect(frame, (x1, y1, w, h), l=4, t=2)
        cvzone.putTextRect(frame, f"ID: {obj_id}", (max(0, x1), max(40, y1)), scale=0.7, thickness=1, offset=2)

        # Initialize the direction state if not already initialized
        if obj_id not in vehicle_directions:
            vehicle_directions[obj_id] = None

        # Count vehicles crossing the line in a single direction
        if tracking_line[0] < cx < tracking_line[2] and tracking_line[1] - 15 < cy < tracking_line[1] + 15:
            if vehicle_directions[obj_id] is None:  # If vehicle hasn't been counted yet
                # Define vehicle direction
                if cy > tracking_line[1]:
                    vehicle_directions[obj_id] = "down"
                else:
                    vehicle_directions[obj_id] = "up"

            # Only count the vehicle when it crosses the line in one direction
            if vehicle_directions[obj_id] == "down" and cy > tracking_line[1]:
                if obj_id not in count:
                    count.append(obj_id)
                    cv2.line(frame, (tracking_line[0], tracking_line[1]), (tracking_line[2], tracking_line[3]), (0, 255, 0), 4)
            
            elif vehicle_directions[obj_id] == "up" and cy < tracking_line[1]:
                if obj_id not in count:
                    count.append(obj_id)
                    cv2.line(frame, (tracking_line[0], tracking_line[1]), (tracking_line[2], tracking_line[3]), (0, 255, 0), 4)

        # Reset vehicle direction when it moves far away from the tracking line
        if cy > tracking_line[1] + 50 or cy < tracking_line[1] - 50:
            vehicle_directions[obj_id] = None

    # Display count
    cvzone.putTextRect(frame, f'Count: {len(count)}', (50, 50), scale=2, thickness=2, offset=5)

    # Show Frame
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
