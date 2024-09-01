import torch
import numpy as np

from pathlib import Path

from torch.utils.data import DataLoader, Subset
from torch import nn
from monai import transforms
from torch.utils.tensorboard import SummaryWriter
from dl_mbl_gut import dataloader_avl, model_asym, train, evaluation

# tensorboard stuff
runname = "3d_asym_avl5e-5 learn extra conv"
runs_path = "/mnt/efs/dlmbl/G-bs/runs/"+runname
logger = SummaryWriter(runs_path)

# batch size and subset of datasets, if desired, otherwise sub = None
batch_size = 9
sub = None

#data directory for dataloading
datadir = '/mnt/efs/dlmbl/G-bs/AvL/'
#transforms for data
transform = transforms.Compose([
        transforms.RandSpatialCrop((56,72,72), random_size = False), #min size for AvL images is 59
        transforms.RandRotate90(prob = 0.75, spatial_axes = (1,2)),
        transforms.RandRotate(prob = 0.1),
        transforms.RandAxisFlip(prob = 0.75),
])
#make datasets for training and validation
train_dataset = dataloader_avl.NucleiDataset(root_dir=datadir, transform = transform, traintestval = 'train')
val_dataset = dataloader_avl.NucleiDataset(root_dir=datadir, transform = transform, traintestval = 'val')

#sub set the datasets for short runs
if sub:
    train_dataset = Subset(train_dataset, range(sub))
    print('train_dataset reduced to ', sub)
    val_dataset = Subset(val_dataset, range(sub))

#put datasets into dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

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

# setup rest of stuff for training loop
scan = tuple(np.arange(0.5,1,0.1))
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005)
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
        loss_function= train.DiceCoefficient(), #torch.nn.BCELoss(),
    )

    torch.save(model.state_dict(), f'/mnt/efs/dlmbl/G-bs/models/{runname}_model_epoch_{epoch+1}.pth')
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

