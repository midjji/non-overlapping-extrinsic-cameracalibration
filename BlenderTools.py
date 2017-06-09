import bpy
import bpy_extras
from mathutils import Matrix, Vector
import numpy as np
import errno
from pathlib import Path

def MakeDir(path):
    try:
        path.mkdir(parents=True)
    except OSError as e:
        if e.errno == errno.EEXIST and path.is_dir():
            pass
        else:
            raise

# This code is from
# http://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera

def GetIntrinsics(cam):
    """ Create a 3x4 P matrix from Blender camera """
    camd = cam.data

    f_in_mm = camd.lens
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = camd.sensor_width
    sensor_height_in_mm = camd.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    if (camd.sensor_fit == 'VERTICAL'):
        # the sensor height is fixed (sensor fit is horizontal),
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else: # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal),
        # the sensor height is effectively changed with the pixel aspect ratio
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm


    # Parameters of intrinsic calibration matrix K
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_v
    u_0 = resolution_x_in_px * scale / 2
    v_0 = resolution_y_in_px * scale / 2
    skew = 0 # only use rectangular pixels

    K = Matrix(
        ((alpha_u, skew,    u_0),
        (    0  , alpha_v, v_0),
        (    0  , 0,        1 )))
    return K


def GetExtrinsics(cam):
    """ Get camera rotation and translation matrices of a Blender camera. """

    # There are 3 coordinate systems involved:
    # 1. The World coordinates: "world"
    #    - right-handed
    # 2. The Blender camera coordinates: "bcam"
    #    - x is horizontal
    #    - y is up
    #    - right-handed: negative z look-at direction
    # 3. The desired computer vision camera coordinates: "cv"
    #    - x is horizontal
    #    - y is down (to align to the actual pixel coordinates 
    #      used in digital images)
    #    - right-handed: positive z look-at direction

    # bcam stands for blender camera
    R_bcam2cv = Matrix(
        ((1, 0,  0),
         (0, -1, 0),
         (0, 0, -1)))

    # Transpose since the rotation is object rotation, 
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam * location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam*cam.location
    # Use location from matrix_world to account for constraints:     
    T_world2bcam = -1*R_world2bcam * location

    # Build the coordinate transform matrix from world to computer vision camera
    R_world2cv = R_bcam2cv*R_world2bcam
    T_world2cv = R_bcam2cv*T_world2bcam

    # put into 3x4 matrix
    RT = Matrix((
        R_world2cv[0][:] + (T_world2cv[0],),
        R_world2cv[1][:] + (T_world2cv[1],),
        R_world2cv[2][:] + (T_world2cv[2],)
         ))
    return RT

def GetCameraProjectionMatrix(cam):

    K = GetIntrinsics(cam)
    RT = GetExtrinsics(cam)
    return K*RT, K, RT

def GetRenderingResolution():

    scene = bpy.context.scene
    s = scene.render.resolution_percentage / 100
    im_w = scene.render.resolution_x * s
    im_h = scene.render.resolution_y * s
    return im_w, im_h

def _NumpyToCvMatrix(matrix, name, indentation=0):
    """ Format a matrix as an YML string.
    :param matrix:  Numpy matrix. dtype=float64 is assumed for now
    :param name:    YML name
    :param indentation: number of spaces to indent
    :return: YML string
    """
    indent = " " * indentation

    s = (indent + "{0}: !!opencv-matrix\n" +
        indent + "   rows: {1}\n" +
        indent + "   cols: {2}\n" +
        indent + "   dt: d\n" +
        indent + "   data: {3}\n").format(name, matrix.shape[0], matrix.shape[1], matrix.flatten().tolist())

    return s

def WriteCamera(cam, cam_filename):
    """ Write camera intrinsics and extrinsics to a .yml file
    :param cam          Blender camera object
    :param cam_filname  Output file
    """
    K = GetIntrinsics(cam)
    P_gt = GetExtrinsics(cam)
    sz = GetRenderingResolution()

    f = open(cam_filename, "w")
    f.write("%YAML:1.0\n---\n")
    f.write("image_size: {0}\n".format([int(sz[0]), int(sz[1])]))
    f.write(_NumpyToCvMatrix(np.array(K), "K"))
    f.write(_NumpyToCvMatrix(np.zeros((1, 5)), "d"))
    f.write(_NumpyToCvMatrix(np.array(P_gt), "P_gt"))
    f.close()

def WriteInfoFile(cam, pattern, fname):
    """ Write chessboard corners, as observed by a camera to an .info.yml file
    :param cam      Blender camera object doing the observations
    :param pattern  ChessboardPattern object
    :param fname    Output file
     """
    flat_corners = []
    for c in pattern.GetInnerCorners2D(cam.name):
        flat_corners.append(c[0] / c[2])
        flat_corners.append(c[1] / c[2])

    f = open(fname, "w")
    f.write("%YAML:1.0\n---\n")
    f.write("pattern_size: [{0}, {1}]\n".format(pattern.inner_cols, pattern.inner_rows))
    f.write("side_length: {0}\n".format(pattern.side_length))
    f.write("chessboard_corners: {0}\n".format(flat_corners))
    f.close()
