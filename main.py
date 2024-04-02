from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO
from calibration.calibrate_camera import load_coefficients, preprocess_image
from tictactoe_engine import infer_tic_tac_toe_state, find_best_move, get_cell_center_and_shorter_edge
from xarm import version
from xarm.wrapper import XArmAPI
from robotic_arm.draw import RobotMain
from robotic_arm.calculate_transformation_matrix import calculate_transformation_matrix, apply_inverse_transformation
from threading import Thread
import base64
from io import BytesIO


try:
    RobotMain.pprint('xArm-Python-SDK Version:{}'.format(version.__version__))
    ARM = XArmAPI('192.168.1.159', baud_checkset=False)
    ROBOT = RobotMain(ARM)
    
except:
    print("yo")

# Load model and calibration data
MTX, DIST, H = load_coefficients("calibration/calibration.yml")
MODEL = YOLO("vision/runs/detect/train2/weights/best.pt")
CLASS_NAMES = ["O", "X", "grid"]
calibration_points = np.array([(267, 165), (251, 279)])
transformed_points = np.array([(0, 0), (0, 120)])
#to transform image coordinate to TCP coordinates
transformation_matrix = calculate_transformation_matrix(calibration_points , transformed_points)

#global bboxes variable
bboxes = None
plan_image = None

app = Flask(__name__)
CORS(app)


def gen_frames(): 
    global bboxes
    global plan_image
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Preprocess the frame
        frame = preprocess_image(frame, MTX, DIST, H, width=560, height=440)
        plan_image=frame
        # Perform YOLO detection
        results = MODEL(frame, stream=False)
        for r in results:
            bboxes = preprocess_bboxes(r.boxes, CLASS_NAMES)
            frame = add_bbox_to_img(frame, bboxes)

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield multipart response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def preprocess_bboxes(bboxes, class_names, conf_threshold=0.5):
    result = {class_name: [] for class_name in class_names}
    for box in bboxes:
        conf = box.conf.item()
        if conf > conf_threshold:
            # Extract xywh and class index
            xywh = box.xywh.cpu().numpy()[0].tolist()  # Convert to numpy, get the first row, convert to list
            class_idx = int(box.cls.item())
            # Add to result
            class_name = class_names[class_idx]
            result[class_name].append(xywh)

    return result


def add_bbox_to_img(img, boxes):
    for cls_name, bboxes in boxes.items():
        for bbox in bboxes:
            # Extract the bounding box coordinates (x, y represent the center of the bbox)
            x, y, w, h = bbox
            x1, y1, x2, y2 = int(x - w/2), int(y - h/2), int(x + w/2), int(y + h/2)
            
            # Draw the bounding box rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            
            # Prepare the text label
            label = f"{cls_name}"  # You can add confidence here if available
            
            # Choose the font, scale, color, and thickness
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2
            
            # Put the text label on the image
            cv2.putText(img, label, (x1, y1), font, fontScale, color, thickness)
    
    return img

def add_points_on_image(img, points, color=(0, 255, 0), size=10, thickness=-1):
    print(type(img))
    for point in points:
        cv2.circle(img, (int(point[0]), int(point[1])), size, color, thickness)
    return img

# Convert the image to a format suitable for JSON serialization
def convert_image_to_base64(img, format='jpeg'):
    # Convert the image to a memory buffer
    is_success, buffer = cv2.imencode(f".{format}", img)
    if not is_success:
        raise ValueError("Could not convert image to buffer")
    # Encode the buffer to base64
    io_buf = BytesIO(buffer)
    base64_str = base64.b64encode(io_buf.getvalue()).decode('utf-8')
    return base64_str

def get_rest_position(bboxes):
    """
    Calculate a rest position for the TCP that is not shadowing or preventing the camera to get the tictactoe grid well
    """
    if 'grid' in bboxes and bboxes['grid']:
        # Assuming the format is [x, y, width, height]
        grid_info = bboxes['grid'][0]
        x, y, width, height = grid_info
        mid_x = x + width / 2
        return (mid_x, 0)
    else:
        return None

def print_grid(grid_state):
    text=""
    for row in grid_state:
        text+='|'.join(row)
        text+='\n'
        text+='---------'
        text+='\n'
    return text

def print_next_move(grid_state, player_letter, move):
    text = f"The best move for player {player_letter} is: {move}"
    grid_state[move[0]][move[1]]=player_letter
    text +=print_grid(grid_state)
    return text


def robot_play(player_letter, transformed_point, shortest_edge, grid_state):
    if player_letter== "X":
        ROBOT.grab_pen((191,43,24),tcp_speed=2)
        ROBOT.draw_x(transformed_point[0], transformed_point[1],  0, shortest_edge/2, tcp_speed=2, tcp_acc=200)
        ROBOT.store_pen((191,43,24), rest_position=(191,0,53),tcp_speed=2)
    else:
        ROBOT.draw_o(transformed_point[0], transformed_point[1],  0, shortest_edge/2, tcp_speed=30, tcp_acc=100)


@app.route('/play', methods=['POST'])
def play():
    if request.method == 'POST':
        try:
            global bboxes
            global plan_image
            grid_state = infer_tic_tac_toe_state(bboxes)
            move, player_letter = find_best_move(grid_state)
            position, shortest_edge = get_cell_center_and_shorter_edge(move,bboxes['grid'][0])
            transformed_point = apply_inverse_transformation(transformation_matrix, [position[0], position[1], 1])
            # Start the robot_play function in a separate thread
            thread = Thread(target=robot_play, args=(player_letter, transformed_point, shortest_edge, grid_state))
            thread.start()
            encoded_image = convert_image_to_base64(add_points_on_image(plan_image, [position], color=(0, 255, 0), size=10, thickness=-1))
            response = {
                "reasoning": {
                    "grid_state": print_grid(grid_state),
                    "next_move": print_next_move(grid_state, player_letter, move)
                },
                "frame": encoded_image
            }
            return jsonify(response), 200
        
        except Exception as e:
            raise e
            return jsonify({"error": str(e)}), 500


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True, port=8000, threaded=True)