import numpy as np

from screen import Screen
from camera import OrthoCamera,PerspectiveCamera
from mesh import Mesh
from renderer import Renderer
from light import PointLight
from animate import Animate



if __name__ == '__main__':
    screen = Screen(500,500)

    camera = OrthoCamera(-1.5, 1.5, -1.5, 1.5, -1.0, -10)
    camera.transform.set_position(0, 0, 2)


    mesh = Mesh.from_stl("unit_cube.stl", np.array([1.0, 0.0, 1.0]),\
        np.array([1.0, 1.0, 1.0]),0.1,1.0,0.2,10)
    mesh.transform.set_rotation(15, 20, 10)

    light = PointLight(50.0, np.array([1, 1, 1]))
    light.transform.set_position(1, 3, 5)

    renderer = Animate(screen, camera, [mesh], light)
    renderer.animate([80,80,80], [0.2, 0.2, 0.2], "rotate", "mlaa")
    #renderer.createFrames([80,80,80], [0.2, 0.2, 0.2], "rotate", "mlaa")
    #renderer.testFrameRate([80,80,80], [0.2, 0.2, 0.2], "rotate", "mlaa")


    screen.show()