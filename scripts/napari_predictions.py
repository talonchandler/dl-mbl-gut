# For viewing random patches
import napari
import numpy as np
import os
from aicsimageio.readers.ome_tiff_reader import OmeTiffReader

preddir = '/mnt/efs/dlmbl/G-bs/AvL/pred/3d_asym_avl_new_augs_model_epoch_6/'
predlist = [x for x in os.listdir(preddir) if x.endswith('.ome.tiff')]

v = napari.Viewer()

# for i in range(len(predlist)):
    # random_index = random.randint(0, len(predlist))
bigpred = OmeTiffReader(preddir + '20240805_488_EGFP-CAAX_640_SPY650-DNA_cell3-02-Subset-02_frame_0.ome.tiff').data[0,:,:,:,:]#preddir + predlist[random_index]).data
    # print(bigpred.shape)
data, mask, pred = bigpred[0], bigpred[1], bigpred[2]
v.add_image(data, name="data")
v.add_labels(mask.astype(np.uint8), name="mask", opacity=0.75)
v.add_image(pred, name="pred", opacity=0.25)
v.add_image(np.max(pred,axis = 0), name="pred", opacity=0.25)
input("Press Enter to continue...")
v.layers.clear()