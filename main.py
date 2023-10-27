from ultralytics import YOLO
import cv2
from calibration.calibrate_camera import calibrate, undistort_image, find_homography, bbox_coordinates_to_world_coordinates, load_coefficients
import matplotlib.pyplot as plt
import math
import numpy as np
import time
import requests
import json


ref_plan="/home/fari/Pictures/calibrationcheckerboard/ref_plan.jpg"
mtx, dist =  load_coefficients("calibration/calibration_coefficients.yml")
ref_plan = undistort_image(ref_plan, mtx, dist)
width=10
height=7
square_size=2.5
H, _ = find_homography(ref_plan, width, height, square_size)
MODEL = YOLO("tictactoe/runs/detect/train5/weights/best.pt")
headers = {'Content-Type': 'application/json'}
url = "http://127.0.0.1:5000/play"
# Create a VideoCapture object





cap = cv2.VideoCapture(0)  # 0 is the default camera

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise Exception("Could not open video device")

# Set properties. Each set returns boolean success code.
# For example, to set the width and height:
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
CLASS_NAMES = ["O", "X", "grid"]

def get_board_state(boxes_world, cls):
    board_state = {}
    for i, box in enumerate(boxes_world):
        class_name = CLASS_NAMES[int(cls[i])]
        if class_name not in board_state:
            board_state[class_name] = []
        board_state[class_name].append(box)
    return board_state


def add_bbox_to_img(img, boxes):
    for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100

            # class name
            cls = int(box.cls[0])

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, CLASS_NAMES[cls], org, font, fontScale, color, thickness)
    return img

N = 90  # Number of frames to calculate the mean number of bounding boxes
bbox_count_history = np.zeros(N, dtype=int)
frame_count = 0
# For FPS calculation
fps = 0
frame_time = time.time()

while True:
    ret, img = cap.read()
    results = MODEL(img, stream=True)
    
    bbox_count = 0  # Initialize bounding box count for this frame

    # coordinates
    for r in results:
        boxes_xywh = r.boxes.xywh.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy()
        bbox_count += len(boxes_xywh)  # Add number of bounding boxes in this result to total count
        boxes_world = [bbox_coordinates_to_world_coordinates(box, H) for box in boxes_xywh]
        img = add_bbox_to_img(img, r.boxes)
        boxes_world_dict = get_board_state(boxes_world, cls)
    
    # Update bounding box count history
    bbox_count_history[frame_count % N] = bbox_count
    frame_count += 1
    
    # Calculate mean number of bounding boxes in the last N frames
    mean_bbox_count = np.mean(bbox_count_history)
    
    # Send API call if mean bounding box count has changed
    if frame_count >= N and mean_bbox_count == bbox_count_history[(frame_count - N) % N] +1: 
        data = {"bounding_boxes": boxes_world_dict}
        json_data = json.dumps(data)
        response = requests.post(url, data=json_data, headers=headers)


    # Calculate FPS
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - frame_time)
    frame_time = new_frame_time
    
    # Display FPS on image
    cv2.putText(img, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



