import numpy as np
from transform import Transform

class PointLight:
    def __init__(self, intensity, color):
        self.transform = Transform()
        
        self.intensity = intensity
        
        self.color = color