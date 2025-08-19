import numpy as np

class Renderer:
    def __init__(self, screen, camera, meshes, light):
        self.screen = screen  
        self.camera = camera  
        self.meshes = meshes  
        self.light = light  

    def render(self, shading, bg_color, ambient_light):
        ambient_light = np.array(ambient_light)

        # Initialize a screen buffer with the background color.
        buffer = np.full((self.screen.height, self.screen.width, 3), bg_color)

        # Create a z-buffer
        z_buffer = np.full((self.screen.width, self.screen.height), -np.inf)

        # Loop through each mesh in the scene.
        for mesh in self.meshes:
            # Transform and project each vertex in the mesh to 2D screen space.
            projected_points = []

            for vertex in mesh.verts:
                # Transform the vertex to camera space.
                transformed_vertex = mesh.transform.apply_to_point(vertex)

                # Project the vertex using the camera.
                p_projected = self.camera.project_point(transformed_vertex)

                # Convert from normalized device coordinates (NDC) to screen coordinates.
                screen_x, screen_y = self.screen.device_to_screen(p_projected[:2])
                projected_points.append((screen_x, screen_y, p_projected[2]))

            # Rasterize each face of the mesh using the projected vertices.
            for face_index, face in enumerate(mesh.faces):
                # Extract the vertices of the face in 2D screen space, including depth.
                face_points = [projected_points[vertex_index] for vertex_index in face]
                world_face_vertices = [mesh.transform.apply_to_point(mesh.verts[v]) for v in face]
                face_normals = [mesh.vertex_normals[v] for v in face]
                v0, v1, v2 = world_face_vertices[:3]
                normal = np.cross(v1 - v0, v2 - v1)
                
                camera_direction = self.camera.transform.apply_inverse_to_point(np.array([0, 0, 0]))
                camera_direction = camera_direction / np.linalg.norm(camera_direction)  # Normalize

                # Cull if the face is facing away from the camera
                if np.dot(normal/np.linalg.norm(normal), camera_direction) > 0:
                    continue
                
                #Compute lighting for phong shading.
                if shading == "phong":
                    self.phong_shade(face_points, world_face_vertices, face_normals, mesh, buffer, z_buffer, ambient_light)

                # Compute lighting for flat shading.
                if shading == "flat":
                    # Compute the light direction in world space.
                    light_position_world = self.light.transform.apply_to_point(np.array([0, 0, 0]))
                    face_center = np.mean([mesh.verts[v] for v in face], axis=0)
                    p=normal
                    L = light_position_world - p
                    L = L / np.linalg.norm(L)

                    N = p/np.linalg.norm(p)

                    camera_position_world = self.camera.transform.apply_to_point(np.array([0,0,0]))
                    V = camera_position_world - p
                    V = V/np.linalg.norm(V)

                    R = L - 2 * np.dot(L,N) * N

                    RVDot = np.power(max(np.dot(R, V), 0), mesh.ke)
                    
                    # Compute ambient lighting.
                    ambient_color = ambient_light * mesh.ka

                    # Compute the dot product for diffuse lighting.
                    diffuse_intensity = max(np.dot(L, N), 0)

                    d = np.linalg.norm(light_position_world - p)

                    # Combine ambient and diffuse lighting.
                    diffuse_color = (mesh.kd * diffuse_intensity * mesh.diffuse_color)

                    specular_color = mesh.specular_color * mesh.ks * RVDot

                    color = ambient_color + diffuse_color + specular_color
                    #color = np.clip(color, 0, 1)
                    color *= 255

                    # Convert face points to arrays for rasterization.
                    a_points = np.array([[p[0], p[1]] for p in face_points])
                    b_points = np.array([[face_points[(i + 1) % len(face_points)][0],
                                        face_points[(i + 1) % len(face_points)][1]] for i in range(len(face_points))])
                    
                    # Fill the polygon on the screen using draw_polygon.
                    self.screen.draw_polygon(
                        a_points,
                        b_points,
                        color=color,
                        buf=buffer,
                        z_buffer=z_buffer,
                        z_points=np.array([[point[2] for point in face_points]])
                    )

                elif shading == "barycentric":
                    face_points = [projected_points[i] for i in face]
                    v0, v1, v2 = face_points[:3]
                    # Assign red, green, blue to each vertex.
                    vertex_colors = np.array([
                        [1.0, 0.0, 0.0],  # Red for the first vertex
                        [0.0, 1.0, 0.0],  # Green for the second vertex
                        [0.0, 0.0, 1.0]   # Blue for the third vertex
                    ])

                    # Compute the bounding box of the triangle to minimize pixel traversal.
                    min_x = max(0, int(min(v0[0], v1[0], v2[0])))
                    max_x = min(self.screen.width - 1, int(max(v0[0], v1[0], v2[0])))
                    min_y = max(0, int(min(v0[1], v1[1], v2[1])))
                    max_y = min(self.screen.height - 1, int(max(v0[1], v1[1], v2[1])))

                    a_points = np.array([[p[0], p[1]] for p in face_points])

                    for y in range(min_y, max_y + 1):
                        for x in range(min_x, max_x + 1):
                            # Calculate barycentric coordinates.
                            alpha, beta, gamma = self.calculate_barycentric((x, y), a_points)

                            # Check if the point is inside the triangle (all barycentric coordinates >= 0).
                            if alpha >= 0 and beta >= 0 and gamma >= 0:
                                # Interpolate depth.
                                z = alpha * v0[2] + beta * v1[2] + gamma * v2[2]

                                # Depth testing.
                                if z > z_buffer[x, y]:
                                    # Interpolate color using barycentric coordinates.
                                    color = alpha * vertex_colors[0] + beta * vertex_colors[1] + gamma * vertex_colors[2]
                                    buffer[x, y] = (color * 255)
                                    z_buffer[x, y] = z



        # Draw the final buffer to the screen.
        self.screen.draw(buffer)

    def phong_shade(self, face_points, world_points, normals, mesh, buffer, z_buffer, ambient_light):
        min_x = max(0, int(min(p[0] for p in face_points)))
        max_x = min(self.screen.width - 1, int(max(p[0] for p in face_points)))
        min_y = max(0, int(min(p[1] for p in face_points)))
        max_y = min(self.screen.height - 1, int(max(p[1] for p in face_points)))

        a_points = np.array([[p[0], p[1]] for p in face_points])

        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                alpha, beta, gamma = self.calculate_barycentric((x, y), a_points)

                if alpha >= 0 and beta >= 0 and gamma >= 0:
                    interpolated_normal = (
                        alpha * np.array(normals[0]) + beta * np.array(normals[1]) + gamma * np.array(normals[2])
                    )
                    interpolated_normal = mesh.transform.apply_to_normal(interpolated_normal)
                    #interpolated_normal /= np.linalg.norm(interpolated_normal)

                    p = alpha * world_points[0] + beta * world_points[1] + gamma * world_points[2]
                    p = interpolated_normal
                    z = p[2]
                    

                    if z > z_buffer[x, y]:
                        # Compute light
                        light_pos = self.light.transform.apply_to_point([0, 0, 0])
                        camera_position_world = self.camera.transform.apply_to_point(np.array([0,0,0]))
                        
                        L = light_pos - p
                        L /= np.linalg.norm(L)


                        # Reflection direction.
                        R = (2 * np.dot(L, interpolated_normal) * interpolated_normal - L)
                        R /= np.linalg.norm(R)

                        
                        V = camera_position_world - p
                        V = V/np.linalg.norm(V)


                        Irrd = (self.light.color * self.light.intensity)/(np.power(np.linalg.norm((light_pos - p)), 2))
                        Refd = np.minimum(1, (mesh.kd * mesh.diffuse_color * np.dot(L, interpolated_normal))/np.pi)

                        # Ambient, diffuse, and specular components.
                        ambient = ambient_light * mesh.ka
                        diffuse = Irrd * Refd
                        specular = mesh.specular_color * np.power(max(0, np.dot(R, V)), mesh.ke) * mesh.ks

                        color = np.clip(ambient + diffuse + specular, 0, 1) * 255
                        #color = (ambient + diffuse + specular)*255

                        buffer[x, y] = color
                        z_buffer[x, y] = z

    def calculate_barycentric(self, p, a):
        A = np.array([a[0, 0], a[1, 0], a[2, 0]])
        B = np.array([a[0, 1], a[1, 1], a[2, 1]])
        #Z = np.array([z_points[0, 0], z_points[0, 1], z_points[0, 2]])
        x = p[0]
        y = p[1]

        denomGamma = ((B[0]-B[1])*A[2] + (A[1]-A[0])*B[2] + A[0]*B[1] - A[1]*B[0])
        denomBeta = ((B[0]-B[2])*A[1] + (A[2]-A[0])*B[1] + A[0]*B[2] - A[2]*B[0])
        if denomGamma == 0 | denomBeta == 0:
            return -1, -1, -1
        gamma = ((B[0]-B[1])*x + (A[1]-A[0])*y + A[0]*B[1] - A[1]*B[0]) / denomGamma
        beta = ((B[0]-B[2])*x + (A[2]-A[0])*y + A[0]*B[2] - A[2]*B[0]) / denomBeta
        alpha = 1 - beta - gamma

        return alpha, beta, gamma
    
    def animate(self, shading, bg_color, ambient_light):
        num = 0
        while num != 6:
            for mesh in self.meshes:
                mesh.verts += np.array([1, 0, 0])
            self.render(shading, bg_color, ambient_light)
            num += 0.1
            print('moved')
