import sys
import numpy as np
from vispy import app, scene, visuals


Scatter3D = scene.visuals.create_visual_node(visuals.MarkersVisual)
canvas = scene.SceneCanvas(keys='interactive', show=True,bgcolor='white')
view = canvas.central_widget.add_view()
view.camera = 'turntable'
view.camera.fov = 45
view.camera.distance = 15

x=np.random.normal(0.0,1.0,1000000)
y=np.random.normal(0.0,1.0,1000000)
z=np.random.normal(0.0,1.0,1000000)

pos = np.stack((x,y,z),axis=1)

p1 = Scatter3D(parent=view.scene)
p1.set_gl_state('translucent', blend=True, depth_test=True)
p1.set_data(pos, face_color='red', symbol='o', size=2,
            edge_width=0.05, edge_color='blue')
print (pos.shape)

if __name__ == '__main__':
    canvas.show()
    if sys.flags.interactive == 0:
        app.run()