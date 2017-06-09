# Stand-alone script for reading a dataset from disk, applying noise and calibrating it.
# The script takes the path to the calibration setup and prints the extrinsics.
# See class CalibrationDataset below for the expected dataset directory layout.

if __name__ == "__main__":
    dset_path = "/home/andreas/test/dataset"

import os
import yaml
import cv2
import numpy as np
from pathlib import Path
from collections import namedtuple
from copy import deepcopy


def RegisterYamlCvMatrix():
    """ Register the OpenCV matrix with the YAML-reader
    Source: https://gist.github.com/autosquid/66acc22b3798b36aea0a
     """
    def opencv_matrix(loader, node):
        mapping = loader.construct_mapping(node, deep=True)
        mat = np.array(mapping["data"])
        mat.resize(mapping["rows"], mapping["cols"])
        return mat

    yaml.add_constructor(u"tag:yaml.org,2002:opencv-matrix", opencv_matrix)


def LoadCvYaml(filename):
    """ Load and parse an OpenCV-generated YAML-file """

    with open(filename) as f:
        s = f.read()
        if s.startswith("%YAML:1.0"):
            s = list(s)
            s[5] = ' '
            s = "".join(s)

        return yaml.load(s)


class CameraNoise:

    def __init__(self,
                 vignetting_gain = 0.25,
                 apply_corner_subpix = True,
                 blur_sigma = 0.5,
                 shot_scale = 0.1,
                 shot_min_sigma = 2,
                 dist_sigma = 0.0,
                 num_samples = 1):
        """
        :param vignetting_gain:     Vignetting gain on left/right edges, relative to the center.
        :param apply_corner_subpix: whether to apply OpenCV's corner subpixel refinement function
        :param blur_sigma:          sigma of lens blur kernel.
        :param shot_scale:          sensor shot noise sigma scale factor
        :param shot_min_sigma:      minimum sensor shot noise standard deviation
        :param dist_sigma:          sigma of residual noise after distortion correction
        :param num_samples:         number of samples to generate with the given parameters
        """

        self.vignetting_gain = vignetting_gain
        self.apply_corner_subpix = apply_corner_subpix
        self.blur_sigma = blur_sigma
        self.shot_scale = shot_scale
        self.shot_min_sigma = shot_min_sigma
        self.dist_sigma = dist_sigma
        self.num_samples = num_samples


class CameraIntrinsics:

    def __init__(self, intrinsics_filename):

        data = LoadCvYaml(intrinsics_filename)
        self.image_size = tuple(data['image_size'])
        self.K = data['K']
        self.d = data['d']

        try:
            self.P_gt = data['P_gt']
            if self.P_gt.shape[0] != 4:
                self.P_gt = np.vstack((self.P_gt, (0, 0, 0, 1)))

        except KeyError:
            self.P_gt = None


class CalibrationPattern:
    """ A set of calibration pattern points and the corresponding image.
        Encapsulates an image and its associated .info.yml-file.
    """
    def __init__(self, info_yml_filename):

        fname = str(info_yml_filename)

        data = LoadCvYaml(fname)
        self.id = os.path.basename(fname)
        self.image_file = fname[:-len(".info.yml")]  # Image file is info_yml file without the ".info.yml" extension
        self.pattern_size = tuple(data['pattern_size'])
        self.side_length = data['side_length']
        self.points = np.array(data['chessboard_corners'], dtype=np.float32).reshape((-1, 2))  # cv2 requires float32
        self.image = cv2.imread(self.image_file, 0).astype(dtype=np.float32)  # Load grayscale image

    def SimulatePhysicalCamera(self, cam_intrinsics, cam_noise, modify=False):
        """ Apply camera noise to the images and detected points in the calibration patterns
        :param cam_intrinsics:  CameraIntrinsics object
        :param cam_noise: CameraNoise object
        :param modify:    [Optional] Modify this CalibrationPattern object if True, return copy if False. Default: False
        :return  CalibrationPattern object with noise applied.
        """

        if modify:
            obj = deepcopy(self)  # Apply noise to a copy of the stored data
        else:
            obj = self  # Apply the noise to the stored data

        # Shorthand aliases
        n = cam_noise
        image = obj.image
        points = obj.points
        h, w = image.shape

        # Apply vignetting
        s = np.arccos(np.power(n.vignetting_gain, 1/4))
        xx, yy = np.meshgrid(np.arange(0, w, 1), np.arange(0, h, 1))

        cx = cam_intrinsics.K[0, 2]
        cy = cam_intrinsics.K[1, 2]

        xx = (xx - cx) / (w/2)
        yy = (yy - cy) / (w/2)
        rr = np.sqrt(xx**2 + yy**2)

        image *= np.cos(rr * s)**4

        # Apply lens blur
        image = cv2.GaussianBlur(image, (5, 5), n.blur_sigma)
        # Apply shot noise and saturation
        image = np.random.normal(image, n.shot_scale * np.sqrt(image) + n.shot_min_sigma)
        image = np.clip(image, 0, 255).astype(dtype=np.uint8)

        # "Detect" pattern corners
        if n.apply_corner_subpix:
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 15, 0.1)  # Refine corners
            cv2.cornerSubPix(image, points, (2, 2), (-1, -1), term)

        # Add distortion correction residual error
        points = np.random.normal(points, n.dist_sigma)

        obj.image = image
        obj.points = points

        return obj

    def Draw(self, image):
        cv2.drawChessboardCorners(image, self.pattern_size, self.points, True)


CameraCalibrationData = namedtuple("CameraCalibrationData", "intrinsics patterns")


class CalibrationDataset:

    def __init__(self, dset_path):
        """
        :param dset_path:  path to dataset
        Expected directory layout:
        ./
            cam0/
                00000.xxx.info.yml - measured points
                00000.xxx          - input image. xxx = {bmp | jpg | png | ...}
                00001.xxx.info.yml
                00001.xxx

            cam1/
                (like cam0)

            cam0.yml - camera 0 intrinsics + ground-truth pose
            cam1.yml - camera 1 intrinsics + ground-truth pose
        """

        self.cameras = []
        self.dset_path = Path(dset_path)

        # Load intrinsics and patterns

        for cam_fname in self.dset_path.glob("*.yml"):
            intrinsics = CameraIntrinsics(str(cam_fname))
            im_dir = cam_fname.with_suffix("")
            patterns = []
            for im_name in im_dir.glob("*.info.yml"):
                patterns.append(CalibrationPattern(im_name))

            self.cameras.append(CameraCalibrationData(intrinsics, patterns))

        # Ensure the patterns are matched by name across cameras

        cam_0 = self.cameras[0]
        for cam_k in self.cameras[1:]:
            assert (len(cam_k.patterns) == len(cam_0.patterns))
            for m_0, m_k in zip(cam_0.patterns, cam_k.patterns):
                assert (m_0.id == m_k.id)

    def SimulatePhysicalCameras(self, cam_noise):
        for cam in self.cameras:
            for m in cam.patterns:
                    m.SimulatePhysicalCamera(cam.intrinsics, cam_noise, modify=True)

    def RemoveImages(self):
        for cam in self.cameras:
            for m in cam.patterns:
                m.image = None

    @staticmethod
    def RelativeTransform(P1, P2):
        """ Compute the relative transform from P1 to P2 """
        from numpy.linalg import inv
        return P2.dot(inv(P1))  # P2 * P1^{-1}


    def Calibrate(self):
        """ Perform extrinsic calibration with the dataset.
        :return: (esimated pose, ground-truth pose, OpenCV calibration RMS reprojection error)
        """

        cam1 = self.cameras[0]
        cam2 = self.cameras[1]

        # Get image points

        im_pts1 = []
        im_pts2 = []
        for m1, m2 in zip(cam1.patterns, cam2.patterns):
            im_pts1.append(m1.points.astype(np.float32))
            im_pts2.append(m2.points.astype(np.float32))

        psz = cam1.patterns[0].pattern_size
        slen = cam1.patterns[0].side_length

        # Generate template pattern points

        w, h = psz
        obj_pts = np.zeros((w * h, 3), np.float32)
        obj_pts[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2) * slen
        obj_pts = [obj_pts] * len(im_pts1)

        # Calibrate

        K1 = cam1.intrinsics.K
        d1 = cam1.intrinsics.d
        K2 = cam2.intrinsics.K
        d2 = cam2.intrinsics.d
        isz = cam1.intrinsics.image_size

        rpe, K1, d1, K2, d2, R, t, E, F = cv2.stereoCalibrate(obj_pts, im_pts1, im_pts2, K1, d1, K2, d2, isz)

        P_est = np.eye(4, 4)
        P_est[0:3, 0:3] = R
        P_est[0:3, 3:4] = t

        P_gt = None
        if cam1.intrinsics.P_gt is not None and cam2.intrinsics.P_gt is not None:
            P_gt = self.RelativeTransform(cam1.intrinsics.P_gt, cam2.intrinsics.P_gt)

        return P_est, P_gt, rpe

RegisterYamlCvMatrix()

if __name__ == "__main__":

    ds = CalibrationDataset(dset_path)
    ds.SimulatePhysicalCameras(CameraNoise())
    P_est, P_gt, rpe = ds.Calibrate()

    print("P_est =", P_est)
    print()
    print("P_gt =", P_gt)
    print()
    print("rpe =", rpe)
