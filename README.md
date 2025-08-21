# 3D-Anti-Aliasing
Simple 3D renderer that tests the framerate of my anti-aliasing methods, all in python!

## Running my program
-To run the program simply run scene.py as shown below  
```
py scene.py
```

## Usage
-This program takes in stl files and adds them into a scene and animates them.  
-There are currently only two options to animate to show that movement works in the engine, horizontal movement and rotation.  
-The only file you should be changing is scene.py, but feel free to change anything else.

## Creating and using mesh files
-Create a mesh using  
```
Mesh.from_stl(<stl_file>, <diffuse_color>, <specular_color>, <ka>, <kd>, <ks>, <ke>)
```
-Diffuse color and specular color should both be in a 1x3 numpy array  
-If you do not know what these color values mean just use my examples or do some research how coloring works with 3D models or play around with it.  
-After creating the your mesh file in scene.py you need to set the rotation and position, by default they will both be at (0,0,0).
```
example_mesh.transform.set_position(x,y,z)
```
```
example_mesh.transform.set_rotation(x,y,z)
```

## Changing other properties
-You can also change properties like your camera and where the light is pointing.  
-This is mainly used for testing so if you know how the values work feel free to change them in scene.py.  
-There are two types of cameras(Perspective and Orthogonal), so feel free to change the camera's class in scene.py to OrthoCamera to see the difference.

## Rendering
-Create your render object using
```
Animate(<screen>, <camera>, [<your_meshes>], <light>)
```
-This is already written in scene.py and the only thing you would ever have to change is the list of meshes being rendered.  
-After you need to call it's animate method
```
your_render.animate(<background_color>, <ambient_light>, <animation_type>, <aliasing>)
```
-background_color and ambient_light should be in 1x3 numpy arrays.  
-Your two options for animation type are 'horizontal' and 'rotate'.
-Your two options for aliasing is 'fxaa' and 'mlaa', anything else it will not apply any anti-aliasing method.    
-Since the puropse of this project is to show the performance of the anti-alaising methods, the meshes themselves are pre-rendered, so starting the program will take awhile. Progress of the rendering will be printed in the console. Though the anti-aliasing will be in real time.  
## Testing framerate
-Using test_framerate will give you the same output as animate, but will print out the framerate in console.
```
your_render.test_framerate(<background_color>, <ambient_light>, <animation_type>, <aliasing>)
```
## Dependencies
--You will of course need python! Any version of python 3 should work, but I used python 3.12.6 if there are any problems.  
--pygame 2.5.0  
--numpy 1.26.4  

## Some example outputs
![alt text](https://github.com/CtrlAltSam/3D-Anti-Aliasing/blob/main/gifs/OrthoFXAA.gif) ![alt_text](https://github.com/CtrlAltSam/3D-Anti-Aliasing/blob/main/gifs/PerspectiveMLAA.gif)  
![alt_text](https://github.com/CtrlAltSam/3D-Anti-Aliasing/blob/main/gifs/SceneMLAA.gif) ![alt_text](https://github.com/CtrlAltSam/3D-Anti-Aliasing/blob/main/gifs/OrthoMLAA.gif)
