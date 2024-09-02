import torch
import numpy as np

from pathlib import Path

from torch.utils.data import DataLoader, Subset
from torch import nn
from monai import transforms
from torch.utils.tensorboard import SummaryWriter
from dl_mbl_gut import dataloader_avl, model_asym, train, evaluation, metrics

# tensorboard stuff
runname = "3d_asym_avl_new_augs_noisecorrectsave"
runs_path = "/mnt/efs/dlmbl/G-bs/runs/"+runname
logger = SummaryWriter(runs_path)

# load a specific state????
model_path = None

# batch size and subset of datasets, if desired, otherwise sub = None
batch_size = 9
sub = None


#data directory for dataloading
datadir = '/mnt/efs/dlmbl/G-bs/AvL/'
#transforms for data
img_transform = transforms.Compose([
        transforms.RandSpatialCrop((56,102,102), random_size = False), #min size for AvL images is 59
        transforms.RandRotate90(prob = 0.75, spatial_axes = (1,2)),
        transforms.RandRotate(prob = 0.1, range_x = np.pi*90/180),
        transforms.CenterSpatialCrop((56,72,72)), #min size for AvL images is 59
        transforms.RandAxisFlip(prob = 0.75),
        transforms.RandScaleIntensityFixedMean(prob=1.0, factors=(0,4)),
        transforms.RandGaussianNoise(prob=0.1, mean=0.0, std=1.0),
])


mask_transform = transforms.Compose([
        transforms.RandSpatialCrop((56,102,102), random_size = False), #min size for AvL images is 59
        transforms.RandRotate90(prob = 0.75, spatial_axes = (1,2)),
        transforms.RandRotate(prob = 0.1, range_x = np.pi*90/180, mode='nearest'),
        transforms.CenterSpatialCrop((56,72,72)), #min size for AvL images is 59
        transforms.RandAxisFlip(prob = 0.75),
])

#make datasets for training and validation
train_dataset = dataloader_avl.NucleiDataset(root_dir=datadir, img_transform = img_transform, mask_transform=mask_transform, traintestval = 'train')
val_dataset = dataloader_avl.NucleiDataset(root_dir=datadir, img_transform = img_transform, mask_transform=mask_transform, traintestval = 'val')

#sub set the datasets for short runs
if sub:
    train_dataset = Subset(train_dataset, range(sub))
    print('train_dataset reduced to ', sub)
    val_dataset = Subset(val_dataset, range(sub))

#put datasets into dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

#number of feature maps
num_fmaps = 64

#assemble model
model = model_asym.UNet(
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
    padding = 'same')

# add a final activation
allmodel = nn.Sequential(
    model,
    nn.Conv3d(num_fmaps, 1, 1),
    nn.Sigmoid()
)

if model_path:
    allmodel.load_state_dict(torch.load(model_path), strict=False)


# setup rest of stuff for training loop
scan = tuple(np.arange(0.5,1,0.1))
optimizer = torch.optim.AdamW(allmodel.parameters(), lr=0.00005)
validation_metric = evaluation.f_beta(beta=1)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
n_epochs = 100
# training loop
for epoch in range(n_epochs):


    train.train(
        allmodel,
        train_dataloader,
        optimizer,
        epoch,
        log_image_interval=20,
        tb_logger=logger,
        device=device,
        loss_function= metrics.DiceCoefficient(), #torch.nn.BCELoss(),
    )

    torch.save(allmodel.state_dict(), f'/mnt/efs/dlmbl/G-bs/models/{runname}_model_epoch_{epoch+1}.pth')
    print(f'Model weights saved for epoch {epoch+1}')

    evaluation.validate(
        allmodel,
        val_dataloader,
        metric=validation_metric,
        step=epoch * len(train_dataloader),
        tb_logger=logger,
        device=device,
        scan = scan,
    )

