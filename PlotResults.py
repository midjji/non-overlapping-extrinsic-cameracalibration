import pickle
from RunExperiments import CalibrationExperiment
from SummarizeExperiments import ExperimentSummary, Unpickle
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt
plt.switch_backend('Qt5Agg')

from ExperimentParameters import dset_root, plots_output_dir

class ErrorGraphData:

    def __init__(self, label):
        self.label = label
        self.sigma = []
        self.dt = []
        self.da = []
        self.rpe = []
        self.H_rpe_mode = []
        self.H_rpe_frequencies = []
        self.H_rpe_bin_centers = []
        self.cal_rpe = []
        pass

    def AddData(self, sigma, dt, da, rpe, rpe_mode, H_rpe_frequencies, H_rpe_centers, cal_rpe):
        self.sigma.append(sigma)
        self.dt.append(dt)
        self.da.append(da)
        self.rpe.append(rpe)
        self.H_rpe_mode.append(rpe_mode)
        self.H_rpe_frequencies.append(H_rpe_frequencies)
        self.H_rpe_bin_centers.append(H_rpe_centers)
        self.cal_rpe.append(cal_rpe)


summaries = Unpickle(dset_root / "summaries.p")
summaries = sorted(summaries, key=lambda s: s.parameters['d_sigma'])
cam_rotations = np.array(sorted(list({s.parameters['rot'] for s in summaries})))
dist_stddevs = np.array(sorted(list({s.parameters['d_sigma'] for s in summaries})))
max_sigma = max(s.parameters['d_sigma'] for s in summaries)
rot_markers = {0.0: '-', 22.5: '.-', 45: 'v-', 90: 's-'}
mode_rot_markers = {22.5: '.:', 45: 'v:', 90: 's:'}

# Generate pose and reprojection errors as function of distortion residual noise ($\epsilon_d$)

error_graph_data = {}

for rot in cam_rotations:
    if rot != 0:
        error_graph_data[rot] = ErrorGraphData("$%.1f^\circ, N = %d$" % (rot, 90 / rot + 1))
    else:
        error_graph_data[rot] = ErrorGraphData("$%.1f^\circ$" % rot)

for s in summaries:
    error_graph_data[s.parameters['rot']].AddData(
        s.parameters['d_sigma'], s.avg_dt, s.avg_da, s.avg_rp_error,
        s.H_rpe_mode, s.H_rpe_frequencies, s.H_rpe_bin_centers, s.avg_cal_rpe)

# Plot translation error

fig_dt = plt.figure()
fsz = 16

lines = []
for rot in cam_rotations:
    egd = error_graph_data[rot]
    L = plt.plot(egd.sigma, egd.dt, rot_markers[rot], label=egd.label)
    lines.extend(L)

plt.legend(handles=lines, loc='upper left', fontsize=fsz*1.3)
plt.tick_params(axis='both', which='major', labelsize=fsz)
plt.tick_params(axis='both', which='minor', labelsize=fsz)
plt.xlim([0, max_sigma])
plt.ylim([0, 1.05 * max(s.avg_dt for s in summaries)])
plt.ylabel("Pose translation error [mm]", fontsize=fsz)
plt.xlabel("Distortion noise $\sigma_d$ [pixels]", fontsize=fsz)

fig_dt.tight_layout()

# Plot rotation error

fig_da = plt.figure()

lines = []
for rot in cam_rotations:
    egd = error_graph_data[rot]
    L = plt.plot(egd.sigma, egd.da, rot_markers[rot], label=egd.label)
    lines.extend(L)

plt.legend(handles=lines, loc='upper left', fontsize=fsz*1.3)
plt.tick_params(axis='both', which='major', labelsize=fsz)
plt.tick_params(axis='both', which='minor', labelsize=fsz)
plt.xlim([0, max_sigma])
plt.ylim([0, 1.05 * max(s.avg_da for s in summaries)])
plt.ylabel("Pose rotation error [degrees]", fontsize=fsz)
plt.xlabel("Distortion noise $\sigma_d$ [pixels]", fontsize=fsz)

fig_da.tight_layout()

# Plot calibration reprojection error means

fig_rpe = plt.figure()

lines = []
for rot in cam_rotations:
    egd = error_graph_data[rot]
    L = plt.plot(egd.sigma, egd.cal_rpe, rot_markers[rot], label=egd.label)
    lines.extend(L)

plt.legend(handles=lines, loc='best', fontsize=fsz)
plt.tick_params(axis='both', which='major', labelsize=fsz)
plt.tick_params(axis='both', which='minor', labelsize=fsz)
plt.xlim([0, max_sigma])
plt.ylim([0, 1.05 * max(s.avg_cal_rpe for s in summaries)])
plt.ylabel("Calibration reprojection error $\epsilon_p$ [pixels]", fontsize=fsz)
plt.xlabel("Distortion noise $\sigma_d$ [pixels]", fontsize=fsz)

fig_rpe.tight_layout()

# Plot reprojection error distributions at different noise levels

#fig_rpe_distr, ax = plt.subplots(1, len(cam_rotations), sharey=True)
#fig_rpe_distr.set_size_inches(18, 6)

for j, rot in enumerate(cam_rotations):
    egd = error_graph_data[rot]

    fig = plt.figure()
    lines = []
    for i, s in enumerate(egd.sigma):
        if s == 0.0 or s == 0.1 or s == 0.2 or s == 0.3 or s == 0.4:
            L = plt.plot(egd.H_rpe_bin_centers[i], egd.H_rpe_frequencies[i], label="$\sigma_d=%.2f$" % s)
            lines.extend(L)

    #fig.semilogy()
    plt.legend(handles=lines, loc='upper right', fontsize=fsz)

    if rot != 0:
        plt.title("%.1f degrees camera-to-camera, N = %d cameras" % (rot, 90 / rot + 1), fontsize=fsz)
    else:
        plt.title("%.1f degrees camera-to-camera" % rot, fontsize=fsz)

    plt.xlabel("Reprojection error $\epsilon_p$ [pixels]", fontsize=fsz)
    plt.ylabel("Frequency", fontsize=fsz)

    plt.ylim([1e-5, 4e-3])
    plt.xlim([5e-2, 4])
    plt.tick_params(axis='both', which='major', labelsize=fsz)
    plt.tick_params(axis='both', which='minor', labelsize=fsz)
    plt.grid(True)
    plt.tight_layout()

    fig.savefig(str(plots_output_dir / ("rpe_distr%02d.pdf" % int(rot))))

# Save

fig_dt.savefig(str(plots_output_dir / "mean_dt.pdf"))
fig_da.savefig(str(plots_output_dir / "mean_da.pdf"))
fig_rpe.savefig(str(plots_output_dir / "cal_rpe.pdf"))

plt.show()
