import torch
from torch.utils.data import Subset, DataLoader
import numpy as np
import pandas as pd
import os
from dl_mbl_gut import model_asym
from dl_mbl_gut.dataloader_avl import NucleiDataset
from aicsimageio.writers.ome_tiff_writer import OmeTiffWriter
from dl_mbl_gut.metrics import DiceCoefficient
from dl_mbl_gut.evaluation import f_beta
from aicsimageio.writers.writer import Writer

batch_size = 9
downsample_factor = [4,8]
scans = np.arange(0.5,1,0.1)

model_name = '3d_asym_avl_new_augs_epoch5correctsave_model_epoch_2'
model_path = f'/mnt/efs/dlmbl/G-bs/models/{model_name}.pth'
input_path = '/opt/dlami/nvme/AvL/'
output_path = f'/mnt/efs/dlmbl/G-bs/AvL/pred/{model_name}/'
if not os.path.exists(output_path):
    os.mkdir(output_path)

num_fmaps = 64

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


modseq = torch.nn.Sequential(
    my_model,
    torch.nn.Conv3d(num_fmaps, 1, 1),
    torch.nn.Sigmoid()
)
modseq.load_state_dict(torch.load(model_path), strict=False)
modseq.eval()
modseq.to("cpu")


# get test data in a dataset
test_data = NucleiDataset(root_dir = input_path, traintestval='test', downsample_factor=downsample_factor)

#do it in batches
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

ds = f_beta(beta = 1)
dslist = []
with torch.inference_mode():
    for imname, padd, test_im, test_mask in Subset(test_data, range(10)):
        padd = np.ceil(padd/2).astype(np.uint8)
        prediction = modseq(test_im.unsqueeze(0))
        fullshape = test_im.squeeze().shape
        maskshape = test_mask.squeeze().shape
        print(padd, padd[0]+maskshape[-3], fullshape, maskshape, test_im.shape, test_mask.shape)
        test_im_ex = np.zeros(np.append([3],np.array(fullshape)))
        test_im_ex[0,:,:,:] = test_im.numpy()
        test_im_ex[1,padd[0]:padd[0]+maskshape[-3],
                   padd[1]:padd[1]+maskshape[-2],
                   padd[2]:padd[2]+maskshape[-1]] = test_mask.numpy()
        test_im_ex[2,:,:,:] = prediction.numpy()
        OmeTiffWriter.save(test_im_ex, output_path + imname + '.ome.tiff')
        print('Saved', imname)
        for s in scans:
            dsmet = ds(test_im_ex[1]>s, test_im_ex[2]>s)
            dslist.append({
                'image': imname,
                'dice_threshold': s,
                'dice_value': dsmet
            })
        print('Recorded metrics for', imname)
# save all of the metrics as a dataframe
df = pd.DataFrame(dslist)
df.to_csv(output_path + 'prediction_metrics.csv')