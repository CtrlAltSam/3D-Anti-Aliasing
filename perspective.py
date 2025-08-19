import numpy as np

from screen import Screen
from camera import PerspectiveCamera,OrthoCamera
from mesh import Mesh
from renderer import Renderer
from light import PointLight
from animate import Animate



if __name__ == '__main__':
    screen = Screen(500,500)

    camera = PerspectiveCamera(-1.0, 1.0, -1.0, 1.0, -1.0, -10)
    camera.transform.set_position(0, 0, 3)


    mesh = Mesh.from_stl("suzanne.stl", np.array([1.0, 0.0, 1.0]),\
        np.array([1.0, 1.0, 1.0]),0.1,1.0,0.2,10)
    mesh.transform.set_rotation(15, 10, 0)
    mesh.transform.set_position(0,0,-3)

    light = PointLight(50.0, np.array([1, 1, 1]))
    light.transform.set_position(1, 3, 5)

    renderer = Animate(screen, camera, [mesh], light)
    renderer.animate([80,80,80], [0.2, 0.2, 0.2], "rotate", "fxaa")
    #renderer.createFrames([80,80,80], [0.2, 0.2, 0.2], "rotate", "fxaa")
    #renderer.testFrameRate([80,80,80], [0.2, 0.2, 0.2], "rotate", "fxaa")

    screen.show()