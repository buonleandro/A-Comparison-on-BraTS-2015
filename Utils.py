import numpy as np
import matplotlib.colors as mcolors

def HexToRGB(value):
    value = value.strip("#")
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def RGBToDec(value):
    return [v/256 for v in value]

def CreateColorMap(hex_list, float_list=None):
    rgb_list = [RGBToDec(HexToRGB(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0, 1, len(rgb_list)))

    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = mcolors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp

maskCMapHexs = ['#000000','#FF8E8D','#8DD88E','#9EC4EA','#FFFF88']
maskCMap = CreateColorMap(maskCMapHexs)