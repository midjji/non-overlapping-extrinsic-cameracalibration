from glob import iglob
from PIL import Image

experiment_data_root = "/media/andreas/Data/calibration-paper/synthetic"

files = list(iglob(experiment_data_root + "/**/*.png", recursive=True))
for i, fname in enumerate(files):
    print("Recompressing image {0} of {1}".format(i+1, len(files)))
    im = Image.open(fname).save(fname)
