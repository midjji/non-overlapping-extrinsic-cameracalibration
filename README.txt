This is a collection of useful tools that can be used to generate and evaluate synthetic camera
calibration datasets. This is "research grade" code so documentation is a bit sparse.

License: MIT

Most code is copyright 2017, Andreas Robinson.
Code marked as coming from stackexchange.com is copyright the original authors and likely
subject to the MIT license as outlined here:
https://meta.stackexchange.com/questions/272956/a-new-code-license-the-mit-this-time-with-attribution-required


To reproduce the paper results, follow the recipe below.

Aquire this software:

 * Linux (e.g Ubuntu 16.04)
 * Blender 2.5+
 * Python 3.x
 * OpenCV (>= v2.4) with Python 3 bindings.
 * Python 3.x packages:
   numpy, joblib, pillow, yaml, matplotlib with the Qt5 backend
   (This is not an exhaustive list, unfortunately, but Python will obviously complain
    if any imported module is missing.)


To create a dataset:

 * Open ExperimentParameters.py in a text editor
 * Edit the 'dset_root' and 'plots_output_dir' paths

 * Download Blender from https://www.blender.org/ and unpack the .tar.gz archive
   in a location of your choice.
 * Start Blender from a terminal. The binary is simply "blender" in the top-level directory.
   Using the terminal lets you see progress and error messages.

 * Within Blender, navigate to and open calibrarion-scene.blend
 * If necessary: Open Blender's text editor, choose 'BlenderStart.py'
 * Click the "Run Script" button in the text editor to start.

The dataset will be written to the 'dset_root' directory in about 5 - 10 minutes.
Note that Blender will be unresponsive until completed. However, the progress can be checked
in the terminal or 'dset_root' directory.


To run the paper experiments:

 * First run

   python3 RunExperiments.py

   from a terminal. Experiments are repeated many times so that robust statistics can be
   collected later. RunExperiments.py can require several hours but will print the estimated time
   remaining as it runs. To make it go faster, open ExperimentParameters.py, find
   CameraNoise.GetAllSets() and reduce M = 1000 to e.g 10 or 100. This will obviously reduce
   impact the results.

 * When the experiments are complete, run

   python3 SummarizeExperiments.py

   to summarize the results. This will also take a while, but is much quicker than the previous step.

 * Finally run

   pythan3 PlotResults.py

   to generate the plots from the paper. The files are written to 'plots_output_dir'.


Files:

ExperimentParameters.py   - all parameters and calibration setups used in the paper experiments.

calibration-scene.blend   - the Blender scene used to generate the synthetic dataset

BlenderStart.py           - Launch script for dataset generation, started from within Blender.
BlenderMain.py            - The dataset generation program "main", invoked by BlenderStart.
BlenderObjects.py         - A collection of functions for creating and manipulating the Blender
                            objects used in the calibration scene.
BlenderTools.py           - A set of tools to extract intrinsic and extrinsic camera parameters and
                            store them in .yml files for further processing.

RunExperiments.py         - Performs the experiments in the paper on the generated dataset.
SummarizeExperiments.py   - Computes reprojection errors and statistics from the experiment results.
PlotResults.py            - Produces plots from the experiment summaries.

Data.py                   - Dataset management classes: CalibrationDataset and Measurement
                            The former encapsulates the dataset, the latter encapsulates one
                            image and the calibration pattern corners observed in that image.
                       
ApplyNoiseAndCalibrate.py - A stand-alone program for reading a synthetic dataset from disk,
                            applying noise and calibrating it. 
                            This was not used to generate any of the paper results, but may be
                            useful as a starting point when studying the dataset file structure.

