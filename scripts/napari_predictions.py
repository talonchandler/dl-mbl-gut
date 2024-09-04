# For viewing random patches
import napari
import random
import numpy as np
import pandas as pd
import os
from aicsimageio.readers.ome_tiff_reader import OmeTiffReader


preddir = '/opt/dlami/nvme/AvL/goodpred/'
foldlist = [x for x in os.listdir(preddir) if '.' not in x]

lowthresh = 0.75
highthresh = 1

####### limit the list to those images below or above a certain metric threshold
dflist = []
for f in foldlist:
    tempdf = pd.read_csv(preddir+f+'/'+'prediction_metrics.csv', index_col = 0)
    dflist.append(tempdf)
df = pd.concat(dflist).reset_index(drop = True)
preddf = df[df.dice_threshold ==0.5].reset_index(drop = True)
preddf = preddf[preddf.dice_value<highthresh].reset_index(drop = True)
preddf = preddf[preddf.dice_value>lowthresh].reset_index(drop = True)



v = napari.Viewer()
print(preddf.columns)
for i in range(len(preddf)):
    random_index = random.randint(0, len(preddf))
    row = preddf.iloc[random_index]
    bigpred = OmeTiffReader(preddir + row.dataset +'/'+ row.image + '.ome.tiff').data[0,:,:,:,:]
    data, mask, pred = bigpred[0], bigpred[1], bigpred[2]
    v.add_image(data, name="data")
    v.add_labels(mask.astype(np.uint8), name="mask", opacity=0.35)
    pred = pred>0.5
    pred = pred.astype(np.uint8)
    pred[pred>0] = 6
    v.add_labels(pred, name="pred", opacity=0.35)
    print(row.dataset, row.dice_value ,row.image)
    input("Press Enter to continue...")
    v.layers.clear()