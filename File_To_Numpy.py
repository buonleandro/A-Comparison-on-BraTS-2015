import os
from imageio import imread
import numpy as np

IMG_N = 110
DATA_FLD = './testing/'
NPY_NAME = "test"

x_out = np.zeros(shape=({}, 512, 512).format(IMG_N), dtype=np.float32)

i = 0

files = os.listdir('{}'.format(DATA_FLD))
print(files)
for f in files:
    name = os.path.splitext(f)[0]
    if (name.__contains__('mri')):
        x_out[i] = imread(os.path.join(DATA_FLD,f))
        i+=1
np.save(os.path.join(DATA_FLD, '{}.npy').format(NAME), x_out)
