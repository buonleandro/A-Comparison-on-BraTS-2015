import os
from Data_Processor import MergeMRIAndSave, ProcessMaskAndSave

root = "./testing/HGG_LGG/"
folder = "testing"
patients = os.listdir(root)
relPaths = [os.path.join(root,x) for x in patients]

for i,x in enumerate(relPaths, start=1):
    files=os.listdir(x)
    flair,t2,t1c,t1,mask = None,None,None,None,None
    for file in files:
        flair = os.path.join(x,file) if file.__contains__('_Flair') else flair
        t2 = os.path.join(x,file) if file.__contains__('_T2') else t2
        t1c = os.path.join(x, file) if file.__contains__('_T1c') else t1c
        t1 = os.path.join(x, file) if file.__contains__('_T1') else t1
        #mask = os.path.join(x, file).replace('\\','/') if file.__contains__('.OT.') else mask
    MergeMRIAndSave(flair, t2, t1c, t1, folder, i)
    #ProcessMaskAndSave(mask, i)
