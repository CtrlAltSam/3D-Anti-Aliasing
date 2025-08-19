import numpy as np

class Transform:
    def __init__(self):
        self.position = np.array([0.0, 0.0, 0.0])
        self.__rotation = np.array([0.0, 0.0, 0.0])
    
    def transformation_matrix(self):
        # Create a 4x4 identity matrix
        matrix = np.identity(4)
        
        # Apply translation
        matrix[0:3, 3] = self.position
        
        # Apply rotation
        rx = np.radians(self.__rotation[0])
        ry = np.radians(self.__rotation[1])
        rz = np.radians(self.__rotation[2])
        
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)
        
        # X Rotation
        rot_x = np.array([[1,  0,   0, 0],
                          [0, cx, -sx, 0],
                          [0, sx,  cx, 0],
                          [0,  0,   0, 1]])
        
        # Y Rotation
        rot_y = np.array([[cy, 0, sy, 0],
                          [ 0, 1,  0, 0],
                          [-sy, 0, cy, 0],
                          [ 0, 0,  0, 1]])

        # Z Rotation
        rot_z = np.array([[cz, -sz, 0, 0],
                          [sz,  cz, 0, 0],
                          [ 0,   0, 1, 0],
                          [ 0,   0, 0, 1]])

        # Combine rotations in XYZ order
        rotation_matrix = rot_x @ rot_y @ rot_z
        
        return matrix @ rotation_matrix

    def set_position(self, x, y, z):
        self.position = np.array([x, y, z])

    def set_rotation(self, x, y, z):
        self.__rotation = np.array([x, y, z])

    def inverse_matrix(self):
        return np.linalg.inv(self.transformation_matrix())

    def apply_to_point(self, p):
        # Convert 3D point to 4D homogeneous coordinates
        p_homogeneous = np.append(p, 1)
        # Apply transformation matrix
        transformed_point = self.transformation_matrix() @ p_homogeneous
        # Return the 3D part
        return transformed_point[:3]

    def apply_inverse_to_point(self, p):
        # Convert 3D point to 4D homogeneous coordinates
        p_homogeneous = np.append(p, 1)
        # Apply inverse transformation matrix
        transformed_point = self.inverse_matrix() @ p_homogeneous
        # Return the 3D part
        return transformed_point[:3]

    def apply_to_normal(self, n):
        # Apply only the rotation part of the transformation matrix to the normal
        rx = np.radians(self.__rotation[0])
        ry = np.radians(self.__rotation[1])
        rz = np.radians(self.__rotation[2])
 

        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)


        #Rotation matrices
        rot_x = np.array([[1,  0,   0],
                          [0, cx, -sx],
                          [0, sx,  cx]])
        
        rot_y = np.array([[cy, 0, sy],
                          [ 0, 1,  0],
                          [-sy, 0, cy]])
        
        rot_z = np.array([[cz, -sz, 0],
                          [sz,  cz, 0],
                          [ 0,   0, 1]])


        # Combine rotations in XYZ order
        rotation_matrix = rot_x @ rot_y @ rot_z

        # Apply rotation matrix to the normal vector
        transformed_normal = rotation_matrix @ n
        # Normalize the resulting normal vector
        return transformed_normal / np.linalg.norm(transformed_normal)