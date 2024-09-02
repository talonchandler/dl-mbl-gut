import torch
import numpy as np

from pathlib import Path
from iohub import open_ome_zarr
from dl_mbl_gut import model, model_asym
from dl_mbl_gut.dataloader_avl import NucleiDataset
from dl_mbl_gut.inference import apply_model_to_zyx_array
from aicsimageio.writers.ome_tiff_writer import OmeTiffWriter

downsample_factor = 8

model_name = 
model_path = f'/mnt/efs/dlmbl/G-bs/models/{model_name}.pth'
input_path = '/mnt/efs/dlmbl/G-bs/data/AvL/raw'
output_path = '/mnt/efs/dlmbl/G-bs/data/AvL/pred/'

# setup model params
my_model = model_asym.UNet(
    in_channels = 1,
    num_fmaps = num_fmaps,
    fmap_inc_factor = 2,
    downsample_factors = [(1,2,2),(2,2,2),(2,2,2)],
    kernel_size_down = [[(1,3,3),(1,3,3),(1,3,3)], [(3,3,3),(3,3,3)], [(3,3,3),(3,3,3)], [(3,3,3),(3,3,3)]],
    kernel_size_up = [[(3,3,3),(3,3,3),(3,3,3)], [(3,3,3),(3,3,3)], [(3,3,3),(3,3,3)]],
    activation = 'ReLU',
    fov = (1, 1, 1),
    voxel_size = (1, 1, 1),
    num_heads = 1,
    constant_upsample = True,
    padding = 'same'
    )
my_model.load_state_dict(torch.load(model_path), strict=False)
my_model.to("cpu")


# get test data in a dataset
test_data = NucleiDataset(root_dir = input_path, traintestval='test', downsample_factor=[6,8])

for crop, test_im, test_mask in test_data:

    prediction = my_model(test_im)
    test_im_ex = np.zeros(np.concat([[3],np.array(test_mask.shape)]))
    test_im_ex[0,crop[0]:,crop[1]:,crop[2]:] = test_im
    test_im_ex[1,:,:,:] = test_mask
    test_im_ex[2,crop[0]:,crop[1]:,crop[2]:] = prediction.numpy()
    OmeTiffWriter.save(prediction.numpy(), output_path
