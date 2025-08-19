import numpy as np
import time
import pygame

class Animate:
    def __init__(self, screen, camera, meshes, light):
        self.screen = screen  
        self.camera = camera  
        self.meshes = meshes  
        self.light = light
    
    #animation options: rotate, horizontal
    #aliasing options: fxxa, mlaa
    def animate(self, bg_color, ambient_light, animation, aliasing):
        # Pre-rendering animation
        frames = self.getFrames(bg_color, ambient_light, animation)
        framesLength = len(frames)
        index = 0
        start_time = time.time()
        running = True
        
        while running:
            #rendering anti-aliasing in real time
            if(aliasing == "fxaa"):
                self.screen.draw(self.fxaa(frames[index]))
            elif(aliasing == "mlaa"):
                self.screen.draw(self.mlaa(frames[index]))
            else:
                self.screen.draw(frames[index])

            index = (index + 1) % framesLength

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Frame limiter (e.g., 60 FPS)
            time.sleep(max(0, 1/22 - (time.time() - start_time)))
            start_time = time.time()
            #print(f"Rotation Angle: {self.rotation_angle}")

        pygame.quit()

    def createFrames(self, bg_color, ambient_light, animation, aliasing):
        running = True
        frames = self.getFrames(bg_color, ambient_light, animation)
        framesLength = len(frames)
        index = 0

        while running:
            if(aliasing == "fxaa"):
                self.screen.draw(self.fxaa(frames[index]))
            elif(aliasing == "mlaa"):
                self.screen.draw(self.mlaa(frames[index]))
            else:
                self.screen.draw(frames[index])

            pygame.image.save(self.screen.screen, f"{index}.png")
            index = (index + 1) % framesLength
            if(index == 0):
                pygame.quit

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False  
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_s:
                        pygame.image.save(self.screen.screen, f"{index}.png")
                    elif event.key == pygame.K_RIGHT:
                        index = (index + 1) % framesLength
                        if(index == 0):
                            print("end")


        pygame.quit()

    def testFrameRate(self, bg_color, ambient_light, animation, aliasing):
        # Pre-rendering animation to look at raw performance of anti-aliasing
        frames = self.getFrames(bg_color, ambient_light, animation)
        framesLength = len(frames)
        index = 0
        running = True
        previous_time = time.time()
        frame_count = 0
        
        while running:
            #rendering anti-aliasing in real time
            if(aliasing == "fxaa"):
                self.screen.draw(self.fxaa(frames[index]))
            elif(aliasing == "mlaa"):
                self.screen.draw(self.mlaa(frames[index]))
            else:
                self.screen.draw(frames[index])
            index = (index + 1) % framesLength
            
            current_time = time.time()
            delta_time = current_time - previous_time
            frame_count += 1

            # Calculate and display FPS every second
            if current_time - previous_time >= 1:
                fps = frame_count / delta_time
                print("FPS:", fps)
                frame_count = 0
                previous_time = current_time

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

        pygame.quit()


    def render(self, bg_color, ambient_light, iteration, animation):
        ambient_light = np.array(ambient_light)

        # Initialize a screen buffer with the background color.
        buffer = np.full((self.screen.height, self.screen.width, 3), bg_color)

        # Create a z-buffer
        z_buffer = np.full((self.screen.width, self.screen.height), -np.inf)

        index = 0

        # Loop through each mesh in the scene.
        for mesh in self.meshes:
            # Transform and project each vertex in the mesh to 2D screen space.
            projected_points = []
            if(animation == "rotate"):
                mesh.transform.set_rotation(0, iteration, 0)

            camera_direction = self.camera.transform.apply_inverse_to_point(np.array([0, 0, 0]))
            camera_direction = camera_direction / np.linalg.norm(camera_direction)  # Normalize
            # Compute light
            light_pos = self.light.transform.apply_to_point([0, 0, 0])
            camera_position_world = self.camera.transform.apply_to_point(np.array([0,0,0]))

            #hard coded moving cube in scene.py
            if(animation == "horizontal" and index == 1):
                mesh_pos = mesh.transform.apply_to_point(np.array([0, 0, 0]))
                mesh_pos[0] += iteration
                mesh.transform.set_position(mesh_pos[0], mesh_pos[1], mesh_pos[2])

            index += 1


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
                #v0, v1, v2 = world_face_vertices[:3]
                #normal = np.cross(v1 - v0, v2 - v1)
                


                # Cull if the face is facing away from the camera
                #if np.dot(normal/np.linalg.norm(normal), camera_direction) >= 0:
                    #continue

                self.phong_shade(face_points, world_face_vertices, face_normals, mesh, buffer, z_buffer, ambient_light, light_pos, camera_position_world)

        # Draw the final buffer to the screen.

        return buffer

    def phong_shade(self, face_points, world_points, normals, mesh, buffer, z_buffer, ambient_light, light_pos, camera_position_world):
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
                    z = p[2]
                    

                    if z > z_buffer[x, y]:

                        
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
        if denomGamma == 0:
            return -1, -1, -1
        gamma = ((B[0]-B[1])*x + (A[1]-A[0])*y + A[0]*B[1] - A[1]*B[0]) / denomGamma
        beta = ((B[0]-B[2])*x + (A[2]-A[0])*y + A[0]*B[2] - A[2]*B[0]) / denomBeta
        alpha = 1 - beta - gamma

        return np.array([alpha, beta, gamma])
    
    def fxaa(self, buffer):
        #height, width, _ = buffer.shape

        # Calculate luminance for the entire buffer
        lum = lambda color: 0.2126 * color[..., 0] + 0.7152 * color[..., 1] + 0.0722 * color[..., 2]
        luminance = lum(buffer)

        # Calculate min and max luminance using neighbors
        padded_luminance = np.pad(luminance, ((1, 1), (1, 1)), mode='edge')
        l_min = np.minimum.reduce([
            padded_luminance[1:-1, :-2],  # left
            padded_luminance[1:-1, 2:],  # right
            padded_luminance[:-2, 1:-1], # up
            padded_luminance[2:, 1:-1],  # down
            luminance                     # center
        ])
        l_max = np.maximum.reduce([
            padded_luminance[1:-1, :-2],  # left
            padded_luminance[1:-1, 2:],  # right
            padded_luminance[:-2, 1:-1], # up
            padded_luminance[2:, 1:-1],  # down
            luminance                     # center
        ])

        # Detect edges
        edge_threshold = 0.125
        edge_mask = (l_max - l_min) >= edge_threshold

        # Average neighboring colors
        padded_buffer = np.pad(buffer, ((1, 1), (1, 1), (0, 0)), mode='edge')
        avg_color = (
            padded_buffer[1:-1, :-2] +  # left
            padded_buffer[1:-1, 2:] +  # right
            padded_buffer[:-2, 1:-1] + # up
            padded_buffer[2:, 1:-1]    # down
        ) / 4

        # Blend edge pixels
        blend_factor = 0.5
        blended_buffer = np.where(edge_mask[..., None], 
                                (1 - blend_factor) * buffer + blend_factor * avg_color, 
                                buffer)

        return np.clip(blended_buffer, 0, 255).astype(np.uint8)
    
    def mlaa(self, buffer):
        #height, width, _ = buffer.shape

        # Convert to grayscale for edge detection
        grayscale = np.dot(buffer[..., :3], [0.299, 0.587, 0.114])

        # Detect edges using Sobel operator
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        grad_x = np.abs(np.convolve(grayscale.flatten(), sobel_x.flatten(), 'same').reshape(grayscale.shape))
        grad_y = np.abs(np.convolve(grayscale.flatten(), sobel_y.flatten(), 'same').reshape(grayscale.shape))
        edge_mask = (grad_x + grad_y) > 100  # Threshold for edges

        # Smooth edges by blending pixel colors
        padded_buffer = np.pad(buffer, ((1, 1), (1, 1), (0, 0)), mode='edge')
        avg_color = (
            padded_buffer[1:-1, :-2] +  # left
            padded_buffer[1:-1, 2:] +  # right
            padded_buffer[:-2, 1:-1] + # up
            padded_buffer[2:, 1:-1]    # down
        ) / 4

        # Apply blending on edge pixels
        blend_factor = 0.5
        blended_buffer = np.where(edge_mask[..., None], 
                                  (1 - blend_factor) * buffer + blend_factor * avg_color, 
                                  buffer)

        return np.clip(blended_buffer, 0, 255).astype(np.uint8)



    
    def getFrames(self, bg_color, ambient_light, animation):
        frames = []
        if(animation == "rotate"):
            rot = 0
            while rot < 360:
                frames.append(self.render(bg_color, ambient_light, rot, "rotate"))
                print(f"{int((rot/360)*100)}%")
                rot += 10
        elif(animation == "horizontal"):
            values = [0, 0.1, 0.1, 0.1, 0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, 0.1, 0.1, 0.1]
            length = len(values)
            index = 0
            for value in values:
                frames.append(self.render(bg_color, ambient_light, value, "horizontal"))
                print(f"{int((index/length)*100)}%")
                index += 1


        return frames

