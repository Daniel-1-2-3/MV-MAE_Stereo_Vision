import pybullet as p
import random

class Shapes:
    def __init__(self):
        self.cube_visual = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[0.4, 0.4, 0.4], 
            rgbaColor=[random.uniform(0.1, 0.9), random.uniform(0.1, 0.9), random.uniform(0.1, 0.9), 1.0])
        
        self.sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.3, 
            rgbaColor=[random.uniform(0.1, 0.9), random.uniform(0.1, 0.9), random.uniform(0.1, 0.9), 1.0])
        
        self.cylinder_visual = p.createVisualShape(shapeType=p.GEOM_CYLINDER, radius=0.3, length=1.0, 
            rgbaColor=[random.uniform(0.1, 0.9), random.uniform(0.1, 0.9), random.uniform(0.1, 0.9), 1.0])
        
        self.cone_visual = p.createVisualShape(shapeType=p.GEOM_CONE, radius=0.35, length=1.5, 
            rgbaColor=[random.uniform(0.1, 0.9), random.uniform(0.1, 0.9), random.uniform(0.1, 0.9), 1.0])
        
        self.capsule_visual = p.createVisualShape(shapeType=p.GEOM_CAPSULE, radius=0.3, length=0.6, 
            rgbaColor=[random.uniform(0.1, 0.9), random.uniform(0.1, 0.9), random.uniform(0.1, 0.9), 1.0])
    
    def get_shapes(self):
        return [[self.cube_visual, self.sphere_visual], 
                [self.capsule_visual, self.cone_visual],    
                [self.cylinder_visual, self.cube_visual],
                ]