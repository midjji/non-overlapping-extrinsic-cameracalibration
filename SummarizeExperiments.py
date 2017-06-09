import sys
import pickle
from pathlib import Path
import numpy as np
from RunExperiments import CalibrationExperiment
from ExperimentParameters import dset_root

def Unpickle(fname):

    import gc
    print("Loading results")

    f = open(str(fname), "rb")
    gc.disable()  # Disabling the garbage collector while loading saves a lot of time
    tests = pickle.load(f)
    gc.enable()
    f.close()

    return tests

class ExperimentSummary:
    """ Summary of a set of experiment runs.
        avg_dt:       average translation error [millimeter]
        avg_da:       average rotation error [degrees]
        avg_cal_rpe:  average calibration reprojection error [RMS, pixels]
        avg_rp_error: average experiment reprojection error [pixels]
        rp_errors:    all experiment reprojection errors [pixels] 
        H_rpe:  histogram of rp_errors
    """

    def __init__(self, tests):
        """
        :param tests: list of CalibrationExperiment() to summarize.
        """

        # Note: Ensure all ex.repeat attributes are the same
        num_cameras = tests[0].repeats + 1
        assert(all(t.repeats == num_cameras-1 for t in tests))

        # Group the experiments and simulate a chain of poses from camera 1 to camera 'num_cameras'
        groups = [tests[i: i + num_cameras] for i in range(0, len(tests), num_cameras)]

        rot_errors = []
        tra_errors = []
        rp_errors = []

        for i, grp in enumerate(groups):

            sys.stderr.write("\rPose %d of %d" % (i, len(groups)))

            P_est = np.eye(4, 4)
            P_gt = np.eye(4, 4)

            # Propagate the pose

            for t in grp:
                P_est = t.P_est.dot(P_est)
                P_gt = t.P_gt.dot(P_gt)

            # Compute errors

            dt, da = CalibrationExperiment.GetCalibrationError(P_est, P_gt)
            cam_int = tests[0].data.cameras[0].intrinsics
            rpe = CalibrationExperiment.SampleReprojectionErrors(cam_int.K, cam_int.image_size, P_est, P_gt)

            rot_errors.append(da)
            tra_errors.append(dt)
            rp_errors.append(rpe)

        self.avg_dt = np.mean(tra_errors) * 1000
        self.avg_da = np.mean(rot_errors) * 180 / np.pi
        self.rp_errors = np.array(rp_errors).flatten()

        self.avg_rp_error = np.mean(self.rp_errors)
        self.avg_cal_rpe = np.mean([t.cal_rp_error for t in tests])

        counts, bins = np.histogram(self.rp_errors, bins=3000, range=(0, 5))
        freqs = counts / np.sum(counts)
        centers = 0.5*(bins[1:] + bins[:-1])

        self.H_rpe_frequencies = freqs
        self.H_rpe_bin_centers = centers
        self.H_rpe_mode = centers[np.argmax(freqs)]

        self.parameters = {}  # Set by the user


## Main ##

if __name__ == "__main__":

    experiments = Unpickle(dset_root / "result.p")

    cam_rotations = np.array(sorted(list({ex.cal_setup.cam_r[2] for ex in experiments})))
    dist_stddevs = np.array(sorted(list({ex.cam_noise.dist_sigma for ex in experiments})))

    summaries = []

    j = 1
    for s in dist_stddevs:
        for i, rot in enumerate(cam_rotations):
            print("%d of %d" % (j, len(cam_rotations) * len(dist_stddevs)))
            j += 1

            r = [ex for ex in experiments if ex.cal_setup.cam_r[2] == rot and ex.cam_noise.dist_sigma == s]
            es = ExperimentSummary(r)
            es.parameters['rot'] = -rot  # Changing the sign here to make it prettier for display
            es.parameters['d_sigma'] = s
            summaries.append(es)

    pickle.dump(summaries, open(str(dset_root / "summaries.p"), "wb"))