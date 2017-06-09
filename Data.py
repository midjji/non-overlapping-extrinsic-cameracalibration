# Utilities and classes managing calibration data (images and their associated .info.yml files)

import os
import yaml
import numpy as np
import cv2
import errno
from pathlib import Path
import pickle
from collections import namedtuple

def MakeDir(path):
    try:
        path.mkdir(parents=True)
    except OSError as e:
        if e.errno == errno.EEXIST and path.is_dir():
            pass
        else:
            raise

# https://gist.github.com/autosquid/66acc22b3798b36aea0a

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

class Measurement:
    """ A set of calibration pattern points and the corresponding image.
        Encapsulates an image and its associated .info.yml-file.
    """
    def __init__(self, info_yml_filename):

        fname = str(info_yml_filename)

        data = LoadCvYaml(fname)
        self.id = os.path.basename(fname)
        self.image_file = fname[:-len(".info.yml")]  # Image file is info_yml file without .info.yml
        self.pattern_size = tuple(data['pattern_size'])
        self.side_length = data['side_length']
        self.gt_points = np.array(data['chessboard_corners'], dtype=np.float32).reshape((-1, 2))  # cv2 requires float32

        self.points = None
        self.image = None

    def SimulatePhysicalCamera(self, cam_noise):
        """ Apply camera noise to the images and detected points.
        Note: This method is somewhat misplaced.
        :param cam_noise: Parameters.CameraNoise() 
        """
        n = cam_noise

        # Load grayscale image
        self.image = cv2.imread(self.image_file, 0).astype(dtype=np.float32)

        # Apply vignetting

        edge_attenuation = 0.25  # Maximum intensity at left/right edge
        s = np.arccos(np.power(edge_attenuation, 1/4))

        h, w = self.image.shape
        cy, cx = h/2, w/2

        xx, yy = np.meshgrid(np.arange(0, w, 1), np.arange(0, h, 1))
        xx = (xx - cx) / (w/2)
        yy = (yy - cy) / (w/2)
        rr = np.sqrt(xx**2 + yy**2)

        self.image *= np.cos(rr * s)**4

        # Apply lens blur
        self.image = cv2.GaussianBlur(self.image, (5, 5), n.blur_sigma)
        # Apply shot noise and saturation
        self.image = np.random.normal(self.image, n.shot_scale * np.sqrt(self.image) + n.shot_min_sigma)
        self.image = np.clip(self.image, 0, 255).astype(dtype=np.uint8)

        # "Detect" pattern corners
        self.points = self.gt_points.copy()
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 15, 0.1)  # Refine corners
        cv2.cornerSubPix(self.image, self.points, (2, 2), (-1, -1), term)

        # Add distortion correction residual error
        self.points = np.random.normal(self.points, n.dist_sigma)

    def Draw(self, image):
        cv2.drawChessboardCorners(image, self.pattern_size, self.points, True)


class IntrinsicParameters:

    def __init__(self, int_params_filename):

        data = LoadCvYaml(int_params_filename)
        self.image_size = tuple(data['image_size'])
        self.K = data['K']
        self.d = data['d']
        self.P_gt = data['P_gt']
        if self.P_gt.shape[0] != 4:
            self.P_gt = np.vstack((self.P_gt, (0, 0, 0, 1)))

CameraCalibrationData = namedtuple("CameraCalibrationData", "intrinsics measurements")

class CalibrationDataset:

    def __init__(self, dset_path):
        """
        :param dset_path:  path to dataset
        Expected directory layout:
        ./
            params.p - pickled ExperimentParameters.CalibrationSetup object (generated in BlenderMain.py)
            cam0/
                00000.xxx.info.yml - measured points
                00000.xxx          - input image. xxx = {bmp | jpg | png | ...}
                00001.xxx.info.yml
                00001.xxx           

            cam1/
                (like cam0)
            cam0.yml - camera 0 intrinsics 
            cam1.yml - camera 1 intrinsics

        :return [int0, int1], [[points0], [points1]] where
        int0, int1 = IntrinsicParameters() and
        [points0], [points1] = lists of MeasuredPatternPoints()
        """

        self.cameras = []
        self.dset_path = Path(dset_path)
        self.cal_setup = pickle.load(open(str(self.dset_path / "parameters.p"), "rb"))

        # Load intrinsics and measured points

        for cam_fname in self.dset_path.glob("*.yml"):
            intrinsics = IntrinsicParameters(str(cam_fname))
            im_dir = cam_fname.with_suffix("")
            measurements = []
            for im_name in im_dir.glob("*.info.yml"):
                measurements.append(Measurement(im_name))

            self.cameras.append(CameraCalibrationData(intrinsics, measurements))

        # Ensure the measurements are matched by name across cameras

        cam_0 = self.cameras[0]
        for cam_k in self.cameras[1:]:
            assert (len(cam_k.measurements) == len(cam_0.measurements))
            for m_0, m_k in zip(cam_0.measurements, cam_k.measurements):
                assert (m_0.id == m_k.id)

    def SimulatePhysicalCamera(self, cam_noise):
        for cam in self.cameras:
            for m in cam.measurements:
                    m.SimulatePhysicalCamera(cam_noise)

    def RemoveImages(self):
        for cam in self.cameras:
            for m in cam.measurements:
                m.image = None

    def RemoveMeasurements(self):
        """ Remove the measurements to reduce loading time when plotting """
        for cam in self.cameras:
            for m in cam.measurements:
                m.gt_points = None
                m.points = None

class DSCache:

    data = {}

    @staticmethod
    def Initialize(dset_root, regenerate_cache=False):
        """ Load and parse the dataset. (Images are not loaded)
        :param dset_root:  pathlib.Path to experiment data root path
        """
        if (dset_root / "cache.p").exists() and not regenerate_cache:
            DSCache.Load(dset_root / "cache.p")
            return

        dset_paths = sorted(Path(dset_root).glob("*"))
        dset_paths = [p for p in dset_paths if p.is_dir()]

        for i, p in enumerate(dset_paths):
            print("Reading dataset %02d of %d: %s" % (i+1, len(dset_paths), p.name))
            DSCache.data[p.name] = CalibrationDataset(p)

        DSCache.Save(dset_root / "cache.p")

    @staticmethod
    def Load(fname):
        print("Loading dataset cache")
        DSCache.data = pickle.load(open(str(fname), "rb"))

    @staticmethod
    def Save(fname):
        print("Saving dataset cache")
        if not DSCache.data:
            raise RuntimeError("Will not save dataset cache - it is empty")
        pickle.dump(DSCache.data, open(str(fname), "wb"))
