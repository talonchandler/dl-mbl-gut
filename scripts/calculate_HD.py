import os
import numpy as np
import pandas as pd
from aicsimageio.readers.ome_tiff_reader import OmeTiffReader
import skimage
from tqdm import tqdm


model_name = '3d_asym_avl_new_augs_noisecorrectsaveepoch3onward_model_epoch_13'
preddir = f'/mnt/efs/dlmbl/G-bs/AvL/pred/{model_name}/'
folds = [x for x in os.listdir(preddir) if '.' not in x]
scans = np.arange(0.3, 1, 0.1)
alldata = []
for f in folds:
    files = [x for x in os.listdir(preddir+f+'/') if x.endswith('.ome.tiff')]
    for fi in tqdm(files):
        curim = OmeTiffReader(preddir+f+'/'+fi).data[0,:,:,:,:]
        for s in scans:
            hdist = skimage.metrics.hausdorff_distance(curim[1,:,:,:]>0,curim[2,:,:,:]>s)
            alldata.append({
                'image':fi.split('.ome.tiff')[0],
                'dataset':f,
                'HD':hdist,
                'threshold':s
            })

df = pd.DataFrame(alldata)
df.to_csv(preddir+'predHD.csv')