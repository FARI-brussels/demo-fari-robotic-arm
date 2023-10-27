import numpy as np

def calculate_transformation_matrix(points, transformed_points):
    # Ensure the input points are numpy arrays for vectorized operations
    points = np.array(points)
    transformed_points = np.array(transformed_points)

    # Calculate differences
    delta_x = points[1, 0] - points[0, 0]
    delta_y = points[1, 1] - points[0, 1]
    delta_x_prime = transformed_points[1, 0] - transformed_points[0, 0]
    delta_y_prime = transformed_points[1, 1] - transformed_points[0, 1]

    # Calculate rotation
    theta = np.arctan2(delta_y_prime, delta_x_prime) - np.arctan2(delta_y, delta_x)

    # Calculate rotation matrix
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])
    
    # Reflection matrix to flip the x-coordinate
    reflection_matrix = np.array([[-1, 0],
                                  [0, 1]])
    
    # Combined rotation and reflection
    combined_transform = np.dot(reflection_matrix, rotation_matrix)

    # Calculate translation
    translation = transformed_points[0] - np.dot(combined_transform, points[0])

    # Create transformation matrix
    transformation_matrix = np.vstack([np.column_stack([combined_transform, translation]),
                                       [0, 0, 1]])
    
    return transformation_matrix


def apply_inverse_transformation(transformation_matrix, point):
    # Calculate the inverse transformation matrix
    inverse_transformation_matrix = np.linalg.inv(transformation_matrix)
    
    # Apply the inverse transformation
    transformed_point = np.dot(inverse_transformation_matrix, point)
    return transformed_point


# Example usage
points = np.array([(199, 276), (197, 443)])
transformed_points = np.array([(0, 0), (0, 160)])
transformation_matrix = calculate_transformation_matrix(points, transformed_points)

print("Transformation Matrix:\n", transformation_matrix)

point_to_transform = np.array([-7, 420, 1])  # The point (-8, 270) with an added 1 for homogeneous coordinates
transformed_point = np.dot(transformation_matrix, point_to_transform)
print("Transformed Point:", transformed_point)


# Apply inverse transformation
inverse_transformed_point = apply_inverse_transformation(transformation_matrix, transformed_point)
print("Inverse Transformed Point:", inverse_transformed_point)
