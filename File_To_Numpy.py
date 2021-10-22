import os
from imageio import imread
import numpy as np

x_out = np.zeros(shape=(110, 512, 512), dtype=np.float32)

DATA_FLD = './testing/'

i = 0

files = os.listdir('{}'.format(DATA_FLD))
print(files)
for f in files:
    name = os.path.splitext(f)[0]
    if (name.__contains__('mri')):
        x_out[i] = imread(os.path.join(DATA_FLD,f))
        i+=1
np.save(os.path.join(DATA_FLD, 'test.npy'), x_out)