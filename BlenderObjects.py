import os
import importlib
os.chdir(os.path.dirname(__file__))

import bpy
import bmesh
import mathutils
from mathutils import Vector, Matrix, Color
from math import pi
from bpy_extras.object_utils import world_to_camera_view

import BlenderTools
importlib.reload(BlenderTools)

def DeleteObjectByName(name):

    obj = bpy.data.objects.get(name)
    if obj is not None:
        mesh = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.meshes.remove(mesh)

def GetRenderingResolution():

    scene = bpy.context.scene
    s = scene.render.resolution_percentage / 100
    im_w = scene.render.resolution_x * s
    im_h = scene.render.resolution_y * s
    return im_w, im_h

def Bootstrap():
    """ Delete all objects in the scene and add the needed objects """

    scene = bpy.context.scene
    data = bpy.data

    for obj in data.objects:
        obj.select = True
    bpy.ops.object.delete()

    # Set scene units and renderer

    scene.render.engine = 'CYCLES'
    scene.unit_settings.system = 'METRIC'
    data.worlds[0].use_nodes = True

    # Add cameras

    cam = data.cameras['Camera']
    cam1 = data.objects.new("Camera1", cam)
    cam2 = data.objects.new("Camera2", cam)

    scene.objects.link(cam1)
    scene.objects.link(cam2)

    SetupCameras()

    # Add light

    lamp_data = data.lamps.new(name="Sun", type='SUN')
    lamp = data.objects.new(name="Sun", object_data=lamp_data)
    scene.objects.link(lamp)
    lamp.location = (0, 0, 6)
    lamp.rotation_mode = 'XYZ'
    lamp.rotation_euler = Vector((27, 0, 0)) / 180.0 * pi

    scene.objects.active = lamp
    # TODO: Set lamp strength to 5.0 from this script.
    # TODO: Set world background color to (0.11, 0.16, 0.44)


def SetupCameras(cam2_r=(90.0, 0.0, 0.0), cam2_t=(1.0, 0.0, 0.0)):
    """ Set camera intrinsics and poses """

    # Set up the camera sensor and lens

    cam = bpy.data.cameras['Camera']
    cam.lens = 1.67  # [mm]
    cam.sensor_fit = 'HORIZONTAL'
    cam.sensor_width = 7.2  # [mm]
    cam.sensor_height = 5.4

    scene = bpy.data.scenes['Scene']
    scene.render.resolution_x = 1600
    scene.render.resolution_y = 1200
    scene.render.resolution_percentage = 100

    # Set up camera poses

    cam1 = bpy.data.objects['Camera1']
    cam1.location = (0, 0, 0)
    cam1.rotation_mode = 'XYZ'
    cam1.rotation_euler = Vector((90, 0, 0)) / 180.0 * pi

    cam2 = bpy.data.objects['Camera2']
    cam2.location = cam2_t  # [meter]
    cam2.rotation_mode = 'XYZ'
    cam2.rotation_euler = Vector(cam2_r) / 180.0 * pi

    return [cam1, cam2]


def DeleteAllPatterns():
    for name, obj in bpy.data.objects.items():
        if name.startswith("pattern"):
            DeleteObjectByName(name)


class ChessboardPattern:

    def __init__(self, obj_name, inner_cols=6, inner_rows=8, side_length=0.12, create_hidden=True):
        """ Create a chessboard pattern
        :param inner_cols: Number of inner corner columns [scalar]
        :param inner_rows: Number of inner corner rows [scalar]
        :param side_length: Square side length [scalar, meter]
        """

        self.obj_name = obj_name
        self.inner_cols = inner_cols
        self.inner_rows = inner_rows
        self.side_length = side_length

        self.verts = []
        self.outline = []  # Vertex indices of outline in clockwise order: [top_left, top_right, bottom_right, bottom_left]
        self.inner = []  # Vertex indices of inner corners, in left-to-right, top-top-bottom order

        inner_cols = self.inner_cols
        inner_rows = self.inner_rows
        side_length = self.side_length

        # Create vertices

        width = (inner_cols + 1) * side_length  # World width
        height = (inner_rows + 1) * side_length # World height

        k = 0
        for j in range(0, inner_rows + 2):
            for i in range(0, inner_cols + 2):

                v = (i * side_length - width/2, j * side_length - height/2, 0)
                self.verts.append(v)

                # if (i == 0 or i == inner_cols + 1) and (j == 0 or j == inner_rows + 1):
                #     # Store indices of points defining the outline
                #     self.outline.append(k)
                # else:
                #     self.inner.append(k)

                if (i != 0 and i != inner_cols + 1) and (j != 0 and j != inner_rows + 1):
                    # Store indices of the inner points
                    self.inner.append(k)
                else:
                    self.outline.append(k)

                k += 1

        # Swap bottom_{left|right} outline points, to get clockwise order
        tmp = self.outline[2]
        self.outline[2] = self.outline[3]
        self.outline[3] = tmp

        self._AddToBlender()
        self.BlenderObject().hide = create_hidden
        self.BlenderObject().hide_render = create_hidden


    def BlenderObject(self):
        """" Get the associated Blender object """
        return bpy.data.objects[self.obj_name]


    def _AddToBlender(self):
        """ Create an associated Blender object """
        # Find or create chessboard materials

        black = bpy.data.materials.get("Black")
        if black is None:
            black = bpy.data.materials.new("Black")
            black.diffuse_color = Color((0, 0, 0))

        white = bpy.data.materials.get("White")
        if white is None:
            white = bpy.data.materials.new("White")
            white.diffuse_color = Color((1.0, 1.0, 1.0))

        # Create faces

        faces = []  # List of tuples of face vertex coordinates

        w = self.inner_cols + 2
        for j in range(0, self.inner_rows + 1):
            for i in range(0, self.inner_cols + 1):
                k = j * w + i
                faces.append((k, k + 1, k + w + 1))
                faces.append((k, k + w + 1, k + w))

        # Create blender object

        mesh = bpy.data.meshes.new('mesh_' + self.obj_name)
        obj = bpy.data.objects.new(self.obj_name, mesh)
        scene = bpy.context.scene
        scene.objects.link(obj)
        scene.objects.active = obj
        mesh.from_pydata(self.verts, [], faces)
        mesh.show_double_sided = True
        mesh.update()

        mesh = obj.data

        # Apply materials to create a checkered pattern

        mesh.materials.append(black)
        mesh.materials.append(white)

        i = 0
        k = 0
        w = 2 * (self.inner_cols + 1)
        for f in mesh.polygons:
            if i % 4 < 2:
                f.material_index = k
            else:
                f.material_index = k ^ 1

            i += 1

            if i == w:
                i = 0
                k ^= 1

        return obj


    def RemoveFromBlender(self):
        """ Delete the Blender associated object """
        DeleteObjectByName(self.obj_name)


    def SetPose(self, rotation_euler, location):
        """
        Place the chessboard pattern. Creates a Blender object if necessary.
        :param rotation_euler:   Rotation around the world X,Y,Z axis in that order. Blender Euler rotation object.
        :param location:         World position. Blender Vector (3D)
        """

        obj = bpy.data.objects.get(self.obj_name)
        if obj is None:
            obj = self._AddToBlender()
        obj.location = location
        obj.rotation_mode = 'XYZ'
        obj.rotation_euler = rotation_euler
        bpy.context.scene.update()

    def _SelectInnerCorners(self):

        """ Select the inner corners - for debugging """

        obj = bpy.data.objects[self.obj_name]
        obj.hide = False
        bpy.context.scene.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')

        mesh = bmesh.from_edit_mesh(bpy.context.object.data)
        mesh.verts.ensure_lookup_table()

        for v in mesh.verts:
            v.select = False

        for i in self.inner:
            mesh.verts[i].select = True

    def _ProjectInto2D(self, cam_name, obj_data_vertices):
        """
        Project pattern vertices into 2D
        :param cam_name:           Name of Blender camera object
        :param obj_data_vertices:  List/array of object vertices.
        :return:  List of image coordinates of the vertices projected into the camera.
        """
        cam = bpy.data.objects[cam_name]
        obj = bpy.data.objects[self.obj_name]
        P, K, RT = BlenderTools.GetCameraProjectionMatrix(cam)

        return [P * obj.matrix_world * v.co for v in obj_data_vertices]

    def GetOutline2D(self, cam_name):
        """
        Get the pattern vertices of the chessboard outine, projected into 2D.
        :param cam_name:  Name of Blender camera object
        :return:  List of image coordinates of the vertices projected into the camera.
        """
        obj = bpy.data.objects[self.obj_name]
        vertices = [obj.data.vertices[i] for i in self.outline]
        return self._ProjectInto2D(cam_name, vertices)

    def GetInnerCorners2D(self, cam_name):
        """
        Get the inner chessboard pattern vertices (i.e the points used in calibration), projected into 2D.
        :param cam_name:  Name of Blender camera object
        :return:  List of image coordinates of the vertices projected into the camera.
        """
        obj = bpy.data.objects[self.obj_name]
        vertices = [obj.data.vertices[i] for i in self.inner]
        return self._ProjectInto2D(cam_name, vertices)

    def AllInnerCornersVisible(self, cam_name, border=5):
        """
        Test whether all inner corners are visible from a camera.
        :param cam_name:  Name of Blender camera object
        :param border:    [Optional] Width of a "keep-out" border, in pixels. Corners within this
                          border are considered to be outside the image. Default is 5.
        :return: True if all inner corners are visible from the camer.
        """
        points = self.GetInnerCorners2D(cam_name)
        im_w, im_h = GetRenderingResolution()
        b = 5  # Border area to avoid
        return all(pt[2] > 0 and b <= pt[0]/pt[2] < im_w-b and b <= pt[1]/pt[2] < im_h-b for pt in points)
