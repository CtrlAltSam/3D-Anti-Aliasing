import numpy as np
import pygame

class Screen:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

        pygame.init()
        self.screen = pygame.display.set_mode([width, height])

    def ratio(self):
        return self.width / self.height
    

    def device_to_screen(self, ndc_coords):
        # NDC point as a homogeneous coordinate
        ndc_point = np.array([ndc_coords[0], ndc_coords[1], 0, 1])

        # Screen transformation matrix
        screen_transform = np.array([
            [self.width / 2, 0, 0, self.width / 2],
            [0, self.height / 2, 0, self.height / 2],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Apply the screen transformation
        screen_point = np.dot(screen_transform, ndc_point)

        # Return integer screen coordinates (x, y)
        return int(screen_point[0]), int(screen_point[1])


    def draw(self, buf: np.ndarray):
        """Takes a buffer of 8-bit RGB pixels and puts them on the canvas.
        buf should be a ndarray of shape (height, width, 3)"""
        # Make sure that the buffer is HxWx3
        if buf.shape != (self.height, self.width, 3):
            raise Exception("buffer and screen not the same size")

        # Flip buffer to account for 0,0 in bottom left while plotting, but 0,0 in top left in pygame
        buf = np.fliplr(buf)

        # The prefered way to accomplish this
        pygame.pixelcopy.array_to_surface(self.screen, buf)

        # An alternative (slower) way, but still valid
        # Iterate over the pixels and paint them
        # for x, row in enumerate(buf):
            # for y, pix in enumerate(row):
                # self.screen.set_at((x, y), pix.tolist())

        # Update the display
        pygame.display.flip()

    def draw_line(self,a,b,color, buf: np.ndarray):

        for i in range(len(a)):
            x1, y1 = a[i]
            x2, y2 = b[i]

            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            sx = 1 if x1 < x2 else -1
            sy = 1 if y1 < y2 else -1
            err = dx - dy

            while True:
                if 0 <= x1 < self.width and 0 <= y1 < self.height:
                    buf[x1, y1] = color

                if x1 == x2 and y1 == y2:
                    break

                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x1 += sx
                if e2 < dx:
                    err += dx
                    y1 += sy

        self.draw(buf)

    #extra credit function
    def draw_polygon(self, a, b, color, buf: np.ndarray, z_buffer: np.ndarray, z_points: np.ndarray):
        """Draw a polygon by filling it in and using depth testing."""
        # Draw the polygon edges
        #for i in range(len(a)):
            #self.draw_line([a[i]], [b[i]], color, buf)

        # Find the bounding box of the polygon
        x_coords = np.concatenate((a[:, 0], b[:, 0]))
        y_coords = np.concatenate((a[:, 1], b[:, 1]))
        min_x, max_x = max(0, np.min(x_coords)), min(self.width, np.max(x_coords))
        min_y, max_y = max(0, np.min(y_coords)), min(self.height, np.max(y_coords))

        # Fill the polygon using a scanline algorithm
        for y in range(min_y, max_y + 1):
            # Find intersections with the polygon edges for this scanline
            intersections = []
            for i in range(len(a)):
                x1, y1 = a[i]
                x2, y2 = b[i]

                # Check if the edge crosses the scanline
                if (y1 <= y < y2) or (y2 <= y < y1):
                    # Calculate the x-coordinate of the intersection
                    if y2 != y1:  # Avoid division by zero
                        x_intersect = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                        intersections.append(int(x_intersect))

            # Sort intersections and fill between pairs
            intersections.sort()
            for i in range(0, len(intersections), 2):
                if i + 1 < len(intersections):
                    x_start = max(min_x, intersections[i])
                    x_end = min(max_x, intersections[i + 1])

                    # Perform depth testing for each pixel
                    for x in range(x_start, x_end + 1):
                        # Interpolate depth values (linear interpolation) for the pixel
                        z_interpolated = self.interpolate_depth(x, y, a, z_points)
                        if z_buffer[x, y] < z_interpolated:  # If the new depth is closer
                            buf[x, y] = color
                            z_buffer[x, y] = z_interpolated  # Update z-buffer

        self.draw(buf)

    def interpolate_depth(self, x, y, a, z_points):
        """Interpolate the depth for a given pixel (x, y) inside the polygon."""
        A = np.array([a[0, 0], a[1, 0], a[2, 0]])
        B = np.array([a[0, 1], a[1, 1], a[2, 1]])
        Z = np.array([z_points[0, 0], z_points[0, 1], z_points[0, 2]])

        denomGamma = ((B[0]-B[1])*A[2] + (A[1]-A[0])*B[2] + A[0]*B[1] - A[1]*B[0])
        denomBeta = ((B[0]-B[2])*A[1] + (A[2]-A[0])*B[1] + A[0]*B[2] - A[2]*B[0])
        if denomGamma == 0 | denomBeta == 0:
            return -np.inf
        gamma = ((B[0]-B[1])*x + (A[1]-A[0])*y + A[0]*B[1] - A[1]*B[0]) / denomGamma
        beta = ((B[0]-B[2])*x + (A[2]-A[0])*y + A[0]*B[2] - A[2]*B[0]) / denomBeta
        alpha = 1 - beta - gamma
        return alpha * Z[0] + beta * Z[1] + gamma * Z[2]

    def show(self):
        """Shows the canvas"""
        running = True
        index = 0
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_s:
                        #from datetime import datetime
                        pygame.image.save(self.screen, "{index}.png")


        pygame.quit()