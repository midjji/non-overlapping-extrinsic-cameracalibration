import os
import time
import pickle
import importlib
os.chdir(os.path.dirname(__file__))
from pathlib import Path
import numpy as np

import bpy
from math import pi
from mathutils import Vector, Euler

import BlenderTools
importlib.reload(BlenderTools)
import BlenderObjects
importlib.reload(BlenderObjects)
import ExperimentParameters
importlib.reload(ExperimentParameters)

from ExperimentParameters import dset_root, plots_output_dir

def SelectVisiblePatterns(patterns, cameras):

    visible_patterns = []
    for p in patterns:

        if all(p.AllInnerCornersVisible(cam.name) for cam in cameras):
            visible_patterns.append(p)
            p.BlenderObject().hide = False
        else:
            p.BlenderObject().hide = True

    return visible_patterns


def AllInnerCornersVisible(pattern, cameras):
    return all(pattern.AllInnerCornersVisible(cam.name) for cam in cameras)


def GenerateFixedPlacements(cameras, params):

    patterns = []

    BlenderObjects.DeleteAllPatterns()

    for i, pose in enumerate(params.GetPatternPoses()):
        p = BlenderObjects.ChessboardPattern("pattern%d" % i)
        p.SetPose(Euler(Vector(pose[0]) / 180 * pi, 'XYZ'), Vector(pose[1]))
        patterns.append(p)

    return patterns

# Set up experiment

def GenerateExperimentDatasets():

    stop_early = -1  # Used for testing, -1 to disable
    skip_render = False  # Set true to only create the patterns

    T_start = time.time()

    print("Generating calibration dataset ...")

    for cal_setup in ExperimentParameters.CalibrationSetup.GetAllSetups():

        print("Calibration setup:", cal_setup.AsString())

        # Create path and store calibration setup

        experiment_path = dset_root / cal_setup.AsString()

        BlenderTools.MakeDir(experiment_path)
        pickle.dump(cal_setup, open(str(experiment_path / "parameters.p"), "wb"))

        # Configure the cameras and generate the patterns

        cameras = BlenderObjects.SetupCameras(cal_setup.cam_r, cal_setup.cam_t)

        if cal_setup.pat_placements in {'c00', 'c22', 'c45', 'c90'}:
            patterns = GenerateFixedPlacements(cameras, cal_setup)
        else:
            raise RuntimeError("Unknown placement")

        bpy.context.scene.update()

        print("Generated %d patterns - now rendering images" % (len(patterns)))

        # Render experiment data

        visible_patterns = SelectVisiblePatterns(patterns, cameras)
        k = 1

        for j, cam in enumerate(cameras):

            cam_path = experiment_path / "cam{}".format(j)
            BlenderTools.WriteCamera(cam, str(cam_path) + ".yml")
            BlenderTools.MakeDir(cam_path)

        for i, p in enumerate(visible_patterns):

            p.BlenderObject().hide_render = False
            p.BlenderObject().hide = False

            bpy.context.scene.update()

            for j, cam in enumerate(cameras):

                cam_path = experiment_path / "cam{}".format(j)
                im_fname = cam_path / "{0:05d}".format(i)
                ext = bpy.context.scene.render.file_extension

                BlenderTools.WriteInfoFile(cam, p, str(im_fname) + ext + ".info.yml")

                bpy.context.scene.render.filepath = str(im_fname)
                bpy.context.scene.camera = cam

                if not skip_render:
                    bpy.ops.render.render(write_still=True)

                k += 1

                if k > stop_early >= 0:
                    return

            p.BlenderObject().hide_render = True

    T_end = time.time();
    T_elapsed = (T_end - T_start) / 60.0

    print("Generation completed in %.1f minutes." % T_elapsed)


BlenderTools.MakeDir(dset_root)
BlenderTools.MakeDir(plots_output_dir)

GenerateExperimentDatasets()
