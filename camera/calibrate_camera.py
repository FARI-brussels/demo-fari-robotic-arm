"""
use it like 
python calibrate_camera.py --image_dir /home/fari/Pictures/calibrationcheckerboard/calibration --image_format jpg --square_size 2.5 --width 10 --height 7 --save_file ./calibration.yml --ref_plan /home/fari/Pictures/calibrationcheckerboard/ref_plan.jpg
"""
import numpy as np
import cv2
import glob
import argparse

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def calibrate(dirpath, image_format, square_size, ref_plan_path, width, height):
    """ Apply camera calibration operation for images in the given directory path. """
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
    objp = np.zeros((height*width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

    #objp = objp * square_size
    objp = objp 

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    if dirpath[-1:] == '/':
        dirpath = dirpath[:-1]

    images = glob.glob(dirpath+'/' + '*.' + image_format)
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (width, height), corners2, ret)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    #find homography
    ref_plan = cv2.imread(ref_plan_path)
    gray = cv2.cvtColor(ref_plan, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (width, height), None)
    print(ret)
    if ret:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        # Define the world coordinates of the checkerboard corners
        # For simplicity, let's assume each square in the checkerboard has a size of 1 unit (e.g., meter)
        world_points = np.zeros((width*height, 2), dtype=np.float32)
        for i in range(width):
            for j in range(height):
                world_points[i*height + j] = [i, j]

        # Compute the homography
        H, _ = cv2.findHomography(corners2 , world_points)

    return [ret, mtx, dist, rvecs, tvecs, H]


def save_coefficients(mtx, dist, H, path):
    """ Save the camera matrix, distortion coefficients and perspective matrix to given path/file. """
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    cv_file.write("K", mtx)
    cv_file.write("D", dist)
    cv_file.write("H", H)
    cv_file.release()

# Also, modify the load_coefficients function to retrieve the perspective matrix
def load_coefficients(path):
    """ Loads camera matrix, distortion coefficients and perspective matrix. """
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    camera_matrix = cv_file.getNode("K").mat()
    dist_matrix = cv_file.getNode("D").mat()
    H = cv_file.getNode("H").mat()
    cv_file.release()
    return [camera_matrix, dist_matrix, H]


def undistort_image(img_path, calibration_file):
    """
    Load the coefficients, undistort and warp the input image.
    
    Parameters:
    - img_path: Path to the image to be undistorted and warped.
    - calibration_file: Path to the YML file with calibration matrices.
    
    Returns:
    - undistorted_img: Undistorted image.
    """
    
    # 1. Load the coefficients
    camera_matrix, dist_coeffs, perspective_matrix = load_coefficients(calibration_file)
    
    # 2. Undistort the image
    img = cv2.imread(img_path)
    undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs)
    return undistorted_img

def transform_bbox_to_world(bbox, H):
    """
    Transforms an image bounding box to a 2D world coordinate bounding box.
    
    Parameters:
    - bbox: A tuple (x, y, w, h) representing the image bounding box.
    - H: The homography matrix.
    
    Returns:
    - A tuple (X_min, Y_min, X_max, Y_max) representing the world bounding box.
    """
    
    x, y, w, h = bbox
    corners_image = np.array([
        [[x, y]],
        [[x + w, y]],
        [[x, y + h]],
        [[x + w, y + h]]
    ], dtype=np.float32)
    
    corners_world = cv2.perspectiveTransform(corners_image, H)
    
    X_coords = corners_world[:, 0, 0]
    Y_coords = corners_world[:, 0, 1]
    
    X_min, X_max = np.min(X_coords), np.max(X_coords)
    Y_min, Y_max = np.min(Y_coords), np.max(Y_coords)
    
    return (X_min, Y_min, X_max, Y_max)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Camera calibration')
    parser.add_argument('--image_dir', type=str, required=True, help='image directory path')
    parser.add_argument('--image_format', type=str, required=True,  help='image format, png/jpg')
    parser.add_argument('--square_size', type=float, required=True, help='chessboard square size') # Made this required
    parser.add_argument('--width', type=int, default=9, help='chessboard width size, default is 9')  # Set default values
    parser.add_argument('--height', type=int, default=6, help='chessboard height size, default is 6') # Set default values
    parser.add_argument('--save_file', type=str, required=True, help='YML file to save calibration matrices')
    parser.add_argument('--ref_plan', type=str, required=True, help='Path to the reference plan image')

    args = parser.parse_args()

    ret, mtx, dist, rvecs, tvecs, H = calibrate(args.image_dir, args.image_format, 
                                                                args.square_size, args.ref_plan, args.width, args.height)
    
    save_coefficients(mtx, dist, H, args.save_file)
    print("Calibration is finished. RMS: ", ret)
    