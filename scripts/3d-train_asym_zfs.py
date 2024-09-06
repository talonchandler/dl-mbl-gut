import torch
import numpy as np

from pathlib import Path

from torch.utils.data import DataLoader, Subset
from torch import nn
from monai import transforms
from torch.utils.tensorboard import SummaryWriter
from dl_mbl_gut import model_asym, train, evaluation, metrics, dataloader
from monai.transforms import RandRotate, Compose, RandGaussianSharpen


import datetime

# tensorboard stuff
runname = run_name = datetime.datetime.now().strftime("%m-%d") + "-3d-unet"
runs_path = "/mnt/efs/dlmbl/G-bs/runs/" + runname
logger = SummaryWriter(runs_path)
learning_rate = 0.00005

# load a specific state????
model_path = None

# batch size and subset of datasets, if desired, otherwise sub = None
batch_size = 3
sub = None

# data directory for dataloading
base_path = Path("/mnt/efs/dlmbl/G-bs/data/")
dataset_path = base_path / Path("all-downsample-8x.zarr")
split_path = base_path / Path("all-downsample-2x-split.csv")
useful_chunk_path = base_path / Path("all-downsample-2x.csv")  # all non-zero data

# transforms for data
transform = Compose(
    [
        RandRotate(range_x=np.pi / 8, prob=0.5, padding_mode="zeros"),
    ]
)

# make datasets for training and validation
train_dataset, val_dataset = [
    dataloader.GutDataset(
        dataset_path,
        split_path,
        data_channel_name="Phase3D",
        z_split_width=23,
        z_stride=3,
        split_mode=split_mode,
        useful_chunk_path=useful_chunk_path,
        transform=transform,
        patch_size=256,
        ndim=2,
        new_annotations=True,
        pos_frac=0.5,
    )
    for split_mode in ["train", "test"]
]


# sub set the datasets for short runs
if sub:
    train_dataset = Subset(train_dataset, range(sub))
    print("train_dataset reduced to ", sub)
    val_dataset = Subset(val_dataset, range(sub))

# put datasets into dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# number of feature maps
num_fmaps = 64

# assemble model
model = model_asym.UNet(
    in_channels=1,
    num_fmaps=num_fmaps,
    fmap_inc_factor=2,
    downsample_factors=[(2, 2, 2), (2, 2, 2), (1, 2, 2)],
    kernel_size_down=[
        [(3, 3, 3), (3, 3, 3)],
        [(3, 3, 3), (3, 3, 3)],
        [(3, 3, 3), (3, 3, 3)],
        [(3, 3, 3), (3, 3, 3)],
    ],
    kernel_size_up=[
        [(3, 3, 3), (3, 3, 3)],
        [(3, 3, 3), (3, 3, 3)],
        [(3, 3, 3), (3, 3, 3)],
    ],
    activation="ReLU",
    fov=(1, 1, 1),
    voxel_size=(1, 1, 1),
    num_heads=1,
    constant_upsample=True,
    padding="same",
)

# add a final activation
allmodel = nn.Sequential(model, nn.Conv3d(num_fmaps, 1, 1), nn.Sigmoid())


##### setup rest of stuff for training loop
scan = tuple(np.arange(0.5, 1, 0.1))
optimizer = torch.optim.AdamW(allmodel.parameters(), lr=learning_rate)
validation_metric = evaluation.f_beta(beta=1)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
n_epochs = int(1e6)

if model_path:
    allmodel.load_state_dict(torch.load(model_path), strict=False)

# training loop
for epoch in range(n_epochs):
    train.train(
        allmodel,
        train_dataloader,
        optimizer,
        epoch,
        log_image_interval=40,
        tb_logger=logger,
        device=device,
        loss_function=metrics.DiceCoefficient(),  # torch.nn.BCELoss(),
    )

    if epoch % 10 == 0:
        torch.save(
            allmodel.state_dict(),
            f"/mnt/efs/dlmbl/G-bs/models/{runname}_model_epoch_{epoch+1}.pth",
        )
        print(f"Model weights saved for epoch {epoch+1}")

        evaluation.validate(
            allmodel,
            val_dataloader,
            metric=validation_metric,
            step=epoch * len(train_dataloader),
            tb_logger=logger,
            device=device,
            scan=scan,
        )
