import numpy as np
from Data_Extractor import ReadMRI
from PIL import Image

def normalize_one_volume(self, volume):
    new_volume = np.zeros(volume.shape)
    location = np.where(volume != 0)
    mean = np.mean(volume[location])
    var = np.std(volume[location])
    new_volume[location] = (volume[location] - mean) / var

    return new_volume

def ShiftIntensity(volume):
    location = np.where(volume != 0)
    minimum = np.min(volume[location])
    maximum = np.max(volume[location])
    std = np.std(volume[location])
    value = np.random.uniform(low=-0.1 * std, high=0.1 * std, size=1)
    volume[location] += value
    volume[location][volume[location] < minimum] = minimum
    volume[location][volume[location] > maximum] = maximum

    return volume

def ScaleIntensity(volume):
    location = np.where(volume != 0)
    new_volume = np.zeros(volume.shape)
    IntensityScale = np.random.uniform(0.9, 1, 1)
    new_volume[location] = volume[location] * IntensityScale
    return new_volume

def MergeMRIAndSave(flair, t2, t1c, t1, idx):
    flair = ReadMRI(flair)
    t2 = ReadMRI(t2)
    t1c = ReadMRI(t1c)
    t1 = ReadMRI(t1)

    flair = np.concatenate([flair, np.zeros((5, 240, 240))], axis=0)
    t2 = np.concatenate([t2, np.zeros((5, 240, 240))], axis=0)
    t1c = np.concatenate([t1c, np.zeros((5, 240, 240))], axis=0)
    t1 = np.concatenate([t1, np.zeros((5, 240, 240))], axis=0)

    flair = ScaleIntensity(flair)
    t1 = ScaleIntensity(t1)
    t2 = ScaleIntensity(t2)
    t1c = ScaleIntensity(t1c)

    flair = ShiftIntensity(flair)
    t1 = ShiftIntensity(t1)
    t2 = ShiftIntensity(t2)
    t1c = ShiftIntensity(t1c)

    out = np.stack([normalize_one_volume(flair), normalize_one_volume(t2), normalize_one_volume(t1c), normalize_one_volume(t1)],axis=0)
    out = np.moveaxis(out,1,-1)

    out = out[out.shape[0] // 2, :, :]
    out = out[:, :, out.shape[2] // 2]

    out = Image.fromarray(out)
    out = out.resize((512,512), resample=Image.LANCZOS)
    out.save("./testing/mri-{}.tif".format(idx))

def ProcessMaskAndSave(mask, idx):

    out = ReadMRI(mask)

    out = out[out.shape[0] // 2, :, :]

    out = Image.fromarray(np.float32(out))
    out = out.resize((512, 512), resample=Image.LANCZOS)
    out.save("./training/mask-{}.tif".format(idx))
