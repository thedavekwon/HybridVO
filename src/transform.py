from constants import *
from PIL import Image

import glob

seq = "00"
files = sorted(glob.glob(f"{KITTI_PATH}/sequences/{seq}/**/*.png"))

print(len(files))

for i, filename in enumerate(files):
    print(i, filename)
    im = Image.open(filename)
    imResize = im.resize((613, 185))
    imResize.save(filename, quality=100)