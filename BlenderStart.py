# This script starts the dataset generation.
# It is also found in the .blend file and should be executed from within Blender.

bootstrap_scene = False  # True to bootstrap, false to render a dataset
# Set true to wipe a scene and reinitialize. Only needed when starting from Blender's default new scene.
# After bootstrapping, you need to manually "use nodes" and set the "Sun" lamp strength to 5.0
# and set the world background to Color(0.11, 0.159, 0.439).
# This has already been done in the included .blend file

import bpy
import os
import sys
import importlib

blend_dir = os.path.dirname(bpy.data.filepath)
if blend_dir not in sys.path:
    sys.path.insert(0, blend_dir)
os.chdir(blend_dir)

if bootstrap_scene:
    import BlenderObjects
    importlib.reload(BlenderObjects)
    BlenderObjects.Bootstrap()
else:
    import BlenderMain
    importlib.reload(BlenderMain)
