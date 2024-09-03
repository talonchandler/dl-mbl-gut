# For viewing random patches
import napari
import numpy as np
import os
from aicsimageio.readers.ome_tiff_reader import OmeTiffReader

preddir = '/mnt/efs/dlmbl/G-bs/AvL/pred/3d_asym_avl_new_augs_epoch5correctsave_model_epoch_2/'
predlist = [x for x in os.listdir(preddir) if x.endswith('.ome.tiff')]

v = napari.Viewer()

for i in range(len(predlist)):
    random_index = random.randint(0, len(predlist))
    bigpred = OmeTiffReader(preddir + predlist[random_index]).data[0,:,:,:,:]
    data, mask, pred = bigpred[0], bigpred[1], bigpred[2]
    v.add_image(data, name="data")
    v.add_labels(mask.astype(np.uint8), name="mask", opacity=0.75)
    v.add_image(pred, name="pred", opacity=0.25)
    input("Press Enter to continue...")
    v.layers.clear()