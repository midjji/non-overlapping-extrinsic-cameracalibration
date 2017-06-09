import os
from copy import deepcopy
from Data import *
from glob import glob
from pathlib import Path
from numpy.linalg import inv, norm
from time import time
from joblib import Parallel, delayed
import multiprocessing
from pyquaternion import Quaternion
import ExperimentParameters

from ExperimentParameters import dset_root

class ExperimentDispatcher:

    def __init__(self, experiments, job_method):
        """ Set up a batch of experiments, aka job
        :param experiments: List of experiment objects
        :param job_method: Method to call on the experiment objects.
               It is expected to return the experiment object itself.
        """
        self.experiments = experiments
        self.job_method = job_method
        # Ensure all runner processes generate different random numbers
        np.random.seed(os.getpid())

    def _Run(self):
        """ Run the batch of experiments
        :return list of completed experiments
        """
        completed_experiments = []
        t0 = time()
        for i, ex in enumerate(self.experiments):

            completed_experiments.append(self.job_method(ex))
            t1 = time()

            # Estimated remaining runtime

            avg_runtime = (t1 - t0) / (i+1)
            tot_runtime = avg_runtime * len(self.experiments)
            rem_runtime = tot_runtime - (t1 - t0)
            print("Estimated time remaining: %.1f minutes" % (rem_runtime / 60))

        return completed_experiments

    @staticmethod
    def Run(experiments, job_method, n_jobs):
        """ Run experiments across multiple processes
        :param experiments: List of experiment objects
        :param job_method: Method to call on the experiment objects,
               this method is expected to return the experiment object itself
        :param n_jobs: Number of processes to distribute the work to 
        :return list of completed experiments
        """
        batches_todo = [[] for i in range(n_jobs)]
        for i, ex in enumerate(experiments):
            batches_todo[i % n_jobs].append(ex)

        jobs = [ExperimentDispatcher(b, job_method) for b in batches_todo]
        batches_done = Parallel(n_jobs=n_jobs)(delayed(ExperimentDispatcher._Run)(job) for job in jobs)

        # flatten the list of batched experiments (read this as two nested for-loops)
        completed_experiments = [ex for batch in batches_done for ex in batch]
        return completed_experiments

class CalibrationExperiment:

    def __init__(self, data, cam_noise, repeats):
        """
        :param data: CalibrationDataset object 
        """

        self.id = -1
        self.num_tests = -1
        self.data = data
        self.cal_setup = data.cal_setup
        self.cam_noise = cam_noise
        self.repeats = repeats

        # Outputs

        self.cal_rp_error = -1.0
        self.calibration_ok = False

        self.P_gt = np.eye(4, 4)
        self.P_est = np.eye(4, 4)
        # self.dt = -1.0
        # self.da = -1.0
        self.rp_errors = []

    @staticmethod
    def RelativeTransform(P1, P2):
        """ Compute the relative transform from P1 to P2 """
        from numpy.linalg import inv
        return P2.dot(inv(P1))  # P2 * P1^{-1}

    @staticmethod
    def RepeatRelativeTransform(P, n):
        P_out = P.copy()
        for i in range(n):
            P_out = P.dot(P_out)

        return P_out

    @staticmethod
    def GetRotationQuaternion(R):

        q = Quaternion()

        tr = R.trace() + 1.0
        if tr > 1e-12:
            S = 0.5 / np.sqrt(tr)
            q[0] = 0.25 / S
            q[1] = (R[2, 1] - R[1, 2]) * S
            q[2] = (R[0, 2] - R[2, 0]) * S
            q[3] = (R[1, 0] - R[0, 1]) * S
            # does not work for a 45 degree rot around y...

        else:
            if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
                S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
                q[0] = (R[2, 1] - R[1, 2]) / S
                q[1] = 0.25 * S
                q[2] = (R[1, 0] + R[0, 1]) / S
                q[3] = (R[0, 2] + R[2, 0]) / S

            elif R[1, 1] > R[2, 2]:
                S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
                q[0] = (R[0, 2] - R[2, 0]) / S
                q[1] = (R[1, 0] + R[0, 1]) / S
                q[2] = 0.25 * S
                q[3] = (R[2, 1] + R[1, 2]) / S

            else:
                S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
                q[0] = (R[1, 0] - R[0, 1]) / S
                q[1] = (R[0, 2] + R[2, 0]) / S
                q[2] = (R[2, 1] + R[1, 2]) / S
                q[3] = 0.25 * S

        error = np.abs(q.norm - 1)

        return q.normalised, error

    @staticmethod
    def GetCalibrationError(P_est, P_gt):
        """ Compute the difference in world translation and rotation
        between the estimated and ground truth camera 
        :return dt = translation error [meter], da = rotation error [radians]
        """

        dP = CalibrationExperiment.RelativeTransform(P_gt, P_est)
        dq, dq_error = CalibrationExperiment.GetRotationQuaternion(dP[0:3, 0:3])
        assert (dq_error < 1.0e-3)
        da = dq.angle

        R_gt = P_gt[0:3,0:3]
        R_est = P_est[0:3, 0:3]
        dt = np.transpose(R_gt).dot(P_gt[0:3,3]) - np.transpose(R_est).dot(P_est[0:3, 3])
        dt = norm(dt)

        return dt, da

    @staticmethod
    def SampleReprojectionErrors(K, isz, P_est, P_gt, n_pts=10000):
        """ Sample the reprojection errors across the image plane of the estimated camera.
        :param K:     Camera intrinsics [3x3 matrix]
        :param isz:   Image size (width, height) [pixels]
        :param P_est: Estimated camera (world-to-camera transform) [4x4 matrix]
        :param P_gt:  Ground truth camera (world-to-camera transform) [4x4 matrix]
        :param n_pts: [optional] number of samples
        :return: vector of error samples 
        """

        from numpy.linalg import inv

        Kinv = np.linalg.inv(K)
        P_gt_inv = np.linalg.inv(P_gt)

        # Place points in random image coordinates at random depths in P_gt

        # Generate random image pixel coordinates
        x = [np.random.uniform((0, 0), isz) for i in range(n_pts)]
        x = np.array(x).transpose()
        x = np.vstack((x, (1.0,) * n_pts))  # homogeneous
        x = Kinv.dot(x)  # homogeneous and normalized

        # Generate random depths [meters]
        z = np.random.uniform(0.1, 100, (1, n_pts)).repeat(3, axis=0)

        # Project into world space and reproject into P_est

        y = x * z  # Move the points away from the image plane into P_gt's space (Hadamard product)
        y = np.vstack((y, (1.0,) * n_pts))  # homogeneous 3D coordinates
        X = P_gt_inv.dot(y)  # Transform into world space
        x2 = P_est.dot(X)  # Reproject into P_est
        x2 = x2[0:3, :] / x2[2, :]

        # Compute reprojection errors

        e_rp = x - x2
        e_rp = e_rp * e_rp  # (Hadamard product)
        e_rp = np.sqrt(e_rp[0, :] + e_rp[1, :])
        e_rp = K[0,0] * e_rp  # pixels

        return e_rp


    def Run(self):

        ds = deepcopy(self.data)

        # Add noise

        ds.SimulatePhysicalCamera(self.cam_noise)

        cam1 = ds.cameras[0]
        cam2 = ds.cameras[1]

        # Get image points

        im_pts1 = []
        im_pts2 = []
        for m1, m2 in zip(cam1.measurements, cam2.measurements):
            im_pts1.append(m1.points.astype(np.float32))
            im_pts2.append(m2.points.astype(np.float32))

        psz = cam1.measurements[0].pattern_size
        slen = cam1.measurements[0].side_length

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

        self.P_gt = self.RelativeTransform(cam1.intrinsics.P_gt, cam2.intrinsics.P_gt)
        ret, K1, d1, K2, d2, R, t, E, F = cv2.stereoCalibrate(obj_pts, im_pts1, im_pts2, K1, d1, K2, d2, isz)

        self.cal_rp_error = ret
        self.P_est[0:3, 0:3] = R
        self.P_est[0:3, 3:4] = t

        # P_gt = self.RepeatRelativeTransform(P_gt, self.repeats)
        # self.P_est = self.RepeatRelativeTransform(self.P_est, self.repeats)
        # self.dt, self.da = self.GetCalibrationError(self.P_est, P_gt)

        # self.rp_errors = self.SampleReprojectionErrors(K1, isz, self.P_est, self.P_gt)

        for m1, m2 in zip(cam1.measurements, cam2.measurements):
            m1.image = None
            m2.image = None

        print("{} of {}:".format(self.id + 1, self.num_tests))
        print(self.cal_setup)
        print(self.cam_noise)

        return self

if __name__ == "__main__":

    DSCache.Initialize(dset_root, regenerate_cache=True)  # Load data

    # Generate experiments

    tests = []
    for dset_name in sorted(DSCache.data.keys()):

        dset = DSCache.data[dset_name]
        pp = dset.cal_setup.pat_placements
        num_repeats = {'c00': 0, 'c22': 3, 'c45': 1, 'c90': 0}

        for cam_noise in ExperimentParameters.CameraNoise.GetAllSets():
            for i in range(cam_noise.num_samples):
                tests.append(CalibrationExperiment(dset, cam_noise, num_repeats[pp]))

    for i, t in enumerate(tests):
        t.id = i
        t.num_tests = len(tests)

    # Run experiments

    t0 = time()

    # completed_tests = [ex.Run() for ex in experiments]
    n_cpus = multiprocessing.cpu_count()
    completed_tests = ExperimentDispatcher.Run(tests, CalibrationExperiment.Run, n_cpus)

    print("Completed in {} seconds".format(time() - t0))

    # Save result

    pickle.dump(completed_tests, open(str(dset_root / "result.p"), "wb"))

    print("Saving result")
    print("Done")
