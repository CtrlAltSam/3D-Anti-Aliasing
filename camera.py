import numpy as np
from transform import Transform

class OrthoCamera:
    def __init__(self, left, right, bottom, top, near, far):
        self.transform = Transform()
        self.left = left
        self.right = right
        self.bottom = bottom
        self.top = top
        self.near = near
        self.far = far
        
        # Orthographic projection matrix
        self.ortho_transform = np.array([
            [2 / (right - left), 0, 0, -(right + left) / (right - left)],
            [0, 2 / (top - bottom), 0, -(top + bottom) / (top - bottom)],
            [0, 0, 2 / (near - far), -(far + near) / (near - far)],
            [0, 0, 0, 1]
        ])

        # Inverse orthographic projection matrix
        self.inverse_ortho_transform = np.array([
            [(right - left) / 2, 0, 0, (right + left) / 2],
            [0, (top - bottom) / 2, 0, (top + bottom) / 2],
            [0, 0, (far - near) / -2, (far + near) / 2],
            [0, 0, 0, 1]
        ])

    def ratio(self):
        return (self.right - self.left) / (self.top - self.bottom)

    def project_point(self, p):
        # Camera transformation
        p_transformed = self.transform.apply_inverse_to_point(p)
        # Convert 3D point to 4D homogeneous coordinate
        p_homogeneous = np.append(p_transformed, 1.0)
        # Apply the orthographic projection
        p_projected = np.dot(self.ortho_transform, p_homogeneous)
        # Return the first 3 elements
        return p_projected[:3]

    def inverse_project_point(self, p):
        p_homogeneous = np.append(p, 1.0)

        p_camera = np.dot(self.inverse_ortho_transform, p_homogeneous)

        p_world = self.transform.apply_to_point(p_camera[:3])
        return p_world
    
class PerspectiveCamera:
    def __init__(self, left, right, bottom, top, near, far):
        self.transform = Transform()  # Initialize the camera transform
        self.left = left
        self.right = right
        self.bottom = bottom
        self.top = top
        self.near = near
        self.far = far
        
        # Perspective projection matrix
        self.perspective_transform = np.dot(np.array([
            [2 / (right - left), 0, 0, -(right + left) / (right - left)],
            [0, 2 / (top - bottom), 0, -(top + bottom) / (top - bottom)],
            [0, 0, 2 / (near - far), -(far + near) / (near - far)],
            [0, 0, 0, 1]
        ]), np.array([
            [near, 0, 0, 0],
            [0, near, 0, 0 ],
            [0, 0, near + far, -far*near],
            [0, 0, 1, 0]
        ]))

        # Inverse perspective projection matrix
        self.inverse_perspective_transform = np.linalg.inv(self.perspective_transform)

    def ratio(self):
        return (self.right - self.left) / (self.top - self.bottom)

    def project_point(self, p):
        # Apply the camera transform to bring the point into camera space
        p_transformed = self.transform.apply_inverse_to_point(p)
        # Convert to homogeneous coordinates for projection
        p_homogeneous = np.append(p_transformed, 1.0)
        # Apply the perspective projection matrix
        p_projected = np.dot(self.perspective_transform, p_homogeneous)
        # Perform perspective division to get NDC coordinates
        p_projected /= p_projected[3]
        return p_projected[:3]

    def inverse_project_point(self, p):
        # Convert point to homogeneous coordinates
        p_homogeneous = np.append(p, 1.0)
        # Apply the inverse perspective matrix to get back to camera space
        p_camera = np.dot(self.inverse_perspective_transform, p_homogeneous)
        # Reverse the perspective division
        p_camera /= p_camera[3]
        # Transform the point from camera space to world space
        p_world = self.transform.apply_to_point(p_camera[:3])
        return p_world