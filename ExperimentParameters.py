import numpy as np
from collections import namedtuple
from pathlib import Path

# Dataset root - the dataset will be created here. This directory will be crated by BlenderMain.py
dset_root = Path("/home/andreas/test/paper-dataset")

# Plot output directory - plots will be written here by PlotResults.py.
#  This directory will be crated by BlenderMain.py
plots_output_dir = Path("/home/andreas/projects/calibration-paper-2017/paper/figures")
#plots_output_dir = Path("/home/andreas/test/plots")

class CameraNoise:

    def __init__(self, blur_sigma, shot_scale, shot_min_sigma, dist_sigma, num_samples):
        """
        :param blur_sigma:      sigma of lens blur kernel
        :param shot_scale:      sensor shot noise sigma scale factor
        :param shot_min_sigma:  minimim sensor shot noise standard deviation
        :param dist_sigma:      sigma of residual noise after distortion correction
        :param num_samples:     number of samples to generate with the given parameters
        """

        self.blur_sigma = blur_sigma
        self.shot_scale = shot_scale
        self.shot_min_sigma = shot_min_sigma
        self.dist_sigma = dist_sigma
        self.num_samples = num_samples

    def __str__(self):
        return "CameraNoise(blur_sigma=%.2f, shot_scale=%.2f, shot_min_sigma=%.2f, dist_sigma=%.2f, num_samples=%d)" % (
            self.blur_sigma, self.shot_scale, self.shot_min_sigma, self.dist_sigma, self.num_samples)

    @staticmethod
    def GetAllSets():
        """ Get all noise parameter sets used in the experiments """
        M = 1000
        return [
            CameraNoise(blur_sigma=0.5, shot_scale=0.1, shot_min_sigma=2, dist_sigma=0.00, num_samples=M),
            CameraNoise(blur_sigma=0.5, shot_scale=0.1, shot_min_sigma=2, dist_sigma=0.05, num_samples=M),
            CameraNoise(blur_sigma=0.5, shot_scale=0.1, shot_min_sigma=2, dist_sigma=0.10, num_samples=M),
            CameraNoise(blur_sigma=0.5, shot_scale=0.1, shot_min_sigma=2, dist_sigma=0.15, num_samples=M),
            CameraNoise(blur_sigma=0.5, shot_scale=0.1, shot_min_sigma=2, dist_sigma=0.20, num_samples=M),
            CameraNoise(blur_sigma=0.5, shot_scale=0.1, shot_min_sigma=2, dist_sigma=0.25, num_samples=M),
            CameraNoise(blur_sigma=0.5, shot_scale=0.1, shot_min_sigma=2, dist_sigma=0.30, num_samples=M),
            CameraNoise(blur_sigma=0.5, shot_scale=0.1, shot_min_sigma=2, dist_sigma=0.35, num_samples=M),
            CameraNoise(blur_sigma=0.5, shot_scale=0.1, shot_min_sigma=2, dist_sigma=0.40, num_samples=M),
            CameraNoise(blur_sigma=0.5, shot_scale=0.1, shot_min_sigma=2, dist_sigma=0.50, num_samples=M),
            CameraNoise(blur_sigma=0.5, shot_scale=0.1, shot_min_sigma=2, dist_sigma=0.60, num_samples=M),
            CameraNoise(blur_sigma=0.5, shot_scale=0.1, shot_min_sigma=2, dist_sigma=0.70, num_samples=M),
        ]


class CalibrationSetup(namedtuple("CalibrationSetup", "pat_placements cam_r cam_t")):
    @staticmethod
    def slugify(s):

        import unicodedata
        import re

        s = str(unicodedata.normalize('NFKD', s).encode('ascii', 'ignore'))
        s = str(re.sub('[^\w=, ]', '', s.strip().lower()))
        return s[1:]

    def AsString(self):
        s = str(self)[len(self.__class__.__name__) + 1 : -1]
        return self.slugify(s)

    def GetPatternPoses(self):

        # 0 degrees between cameras (i.e rectified stereo)
        if self.pat_placements == 'c00':

            cx, cy = 0.4, 2

            poses = [
                ((-90, 0, 9), (cx - 1.6, cy,  -0.1)),
                ((-90, 0, -6),  (cx - 0.7, cy, 0.05)),
                ((-90, 0, 3), (cx, cy, 0.0)),
                ((-90, 0, -10), (cx + 0.5, cy, -0.05)),
                ((-90, 0, -15), (cx + 1.2, cy, 0.1)),
            ]

        # 22.5 degrees between cameras
        elif self.pat_placements == 'c22':

            ad = 11.25
            r = 1.5
            cx, cy = 0.285, -0.055  # Midpoint between the cameras
            w, h = 1.4, 1.25

            ar = ad * np.pi / 180
            ca, sa = np.cos(ar), np.sin(ar)
            dy, dx = sa * w, ca * w  # Tangent direction
            cx += r * sa
            cy += r * ca

            poses = [
                ((-90, 0, -ad), (cx - dx, cy + dy, -h)),
                ((-90, 0, -ad), (cx + dx, cy - dy, -h)),
                ((-90, 0, -ad), (cx, cy, 0)),
                ((-90, 0, -ad), (cx + dx, cy - dy, h)),
                ((-90, 0, -ad), (cx - dx, cy + dy, h)),
            ]

        # 45 degrees between cameras
        elif self.pat_placements == 'c45':

            ad = 22.5
            r = 1.5
            cx, cy = 0.53, -0.22  # Midpoint between the cameras
            w, h = 0.45, 0.92

            ar = ad * np.pi / 180
            ca, sa = np.cos(ar), np.sin(ar)
            dy, dx = sa * w, ca * w  # Tangent direction
            cx += r*sa
            cy += r*ca

            poses = [
                ((-90, 0, -ad), (cx - dx, cy + dy, -h)),
                ((-90, 0, -ad), (cx + dx, cy - dy, -h)),
                ((-90, 0, -ad), (cx, cy, 0)),
                ((-90, 0, -ad), (cx + dx, cy - dy, h)),
                ((-90, 0, -ad), (cx - dx, cy + dy, h)),
            ]

        # 90 degrees between cameras
        elif self.pat_placements == 'c90':

            ad = 45
            h = -1.15
            cx, cy = 3.5, 2.0

            poses = [
                ((-90, 0, -ad), (cx, cy, -2 * h)),
                ((-90, 0, -ad), (cx, cy, -h)),
                ((-90, 0, -ad), (cx, cy, 0)),
                ((-90, 0, -ad), (cx, cy, h)),
                ((-90, 0, -ad), (cx, cy, 2 * h))
            ]

        else:
            raise RuntimeError("placements=\"%s\" is undefined" % self.pat_placements)

        return poses

    @staticmethod
    def GetAllSetups():
        """ Get all physical camera and pattern setups used in the experiments """

        # Paper camera setups
        ex_params = [
            ('c22', (90, 0, -22.5), (0.57, -0.11, 0)),
            ('c45', (90, 0, -45),   (1.06, -0.44, 0)),
            ('c90', (90, 0, -90),   (1.50, -1.50, 0)),
        ]

        # Stereo camera setup
        # ex_params = [
        #     ('c00', (90, 0, 0), (0.40, 0.00, 0))
        # ]

        return [CalibrationSetup(*p) for p in ex_params]

