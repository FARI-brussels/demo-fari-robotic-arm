import cv2
import numpy as np


def segment_grid_into_cells(lines):
    # Separate the lines into horizontal and vertical
    horizontal_lines = []
    vertical_lines = []

    for line in lines:
        rho, theta = line[0]
        if abs(np.sin(theta)) < 0.1:  # Horizontal lines
            horizontal_lines.append(rho)
        else:  # Vertical lines
            vertical_lines.append(rho)

    # Sort the lines
    horizontal_lines.sort()
    vertical_lines.sort()

    # Get intersection points
    intersections = []
    for y in horizontal_lines:
        for x in vertical_lines:
            intersections.append((x, y))

    # Group intersections into cells
    cells = []
    for i in range(2):
        for j in range(2):
            top_left = intersections[i * 3 + j]
            bottom_right = intersections[(i + 1) * 3 + (j + 1)]
            cell = img[int(top_left[1]):int(bottom_right[1]), int(top_left[0]):int(bottom_right[0])]
            cells.append(cell)

    return cells

# Read image
img = cv2.imread('board.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', gray)
# Preprocessing
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
cv2.imshow('blurred', gray)
edges = cv2.Canny(blurred, 50, 150)
cv2.imshow('canny', gray)

# Detect lines
lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
# (Note: Add code to detect grid intersections and segment the image)
print(lines)
# Analyze each cell to determine its state
cells = segment_grid_into_cells(lines)
for cell in cells:
    avg_pixel = np.mean(cell)
    if avg_pixel < threshold:  # Adjust threshold based on your needs
        # Find contours
        contours, _ = cv2.findContours(cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Identify 'X' or 'O' based on contour properties
        # (Note: Add your logic here)

# Display result
cv2.imshow('Processed Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Assuming 'lines' are the detected lines using Hough transform.

