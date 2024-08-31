import torch
import numpy as np

from pathlib import Path

from torch.utils.data import DataLoader
from torch.nn import Sigmoid
from monai import transforms
from torch.utils.tensorboard import SummaryWriter
from dl_mbl_gut import dataloader_avl, model, train, evaluation

# tensorboard stuff
runname = "3davl-test-val-split"
runs_path = "/mnt/efs/dlmbl/G-bs/runs/"+runname
logger = SummaryWriter(runs_path)

#data directory for dataloading
datadir = '/mnt/efs/dlmbl/G-bs/AvL/'
#transforms for data
transform = transforms.Compose([
        transforms.RandRotate90(prob = 0.33, spatial_axes = (1,2)),
        transforms.RandSpatialCrop((56,56,56), random_size = False), #min size for AvL images is 59
        transforms.RandRotate(prob = 0.1),
        transforms.RandAxisFlip(prob = 0.75),
])

train_dataset = dataloader_avl.NucleiDataset(root_dir=datadir, transform = transform, traintestval = 'train')
val_dataset = dataloader_avl.NucleiDataset(root_dir=datadir, transform = transform, traintestval = 'val')

train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=10, shuffle=False)

model = model.UNet(
    depth=3,
    in_channels=1,
    out_channels=1,
    kernel_size=3,
    num_fmaps=64,
    fmap_inc_factor=2,
    padding="valid",
    final_activation=Sigmoid(),
    ndim=3,
)

validation_metric = evaluation.f_beta(beta=1)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
n_epochs = 10
for epoch in range(n_epochs):
    train.train(
        model,
        train_dataloader,
        epoch,
        tb_logger=logger,
        device=device,
        loss_function=torch.nn.BCELoss(),
    )
    
    evaluation.validate(
        model,
        val_dataloader,
        metric=validation_metric,
        step=epoch * len(train_dataloader),
        tb_logger=logger,
        device=device,
    )
