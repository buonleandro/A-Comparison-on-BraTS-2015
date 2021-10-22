import SimpleITK as sitk


def ReadMRI(pathToMHA):

    img = sitk.ReadImage(pathToMHA)

    nda = sitk.GetArrayFromImage(img)

    return nda