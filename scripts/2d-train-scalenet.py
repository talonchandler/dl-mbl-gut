import torch
import numpy as np

from iohub import open_ome_zarr
from pathlib import Path

from torch.utils.data import DataLoader
from torch.nn import Sigmoid
from monai.transforms import RandRotate
from torch.utils.tensorboard import SummaryWriter
from dl_mbl_gut import dataloader, model, train, evaluation, scalenet
import datetime

# Parameters
n_epochs = 100
batch_size = 20
learning_rate = 1e-6

# Input paths
base_path = Path("/mnt/efs/dlmbl/G-bs/data/")
dataset_path_high = base_path / Path("all-downsample-8x.zarr")
dataset_path_low = base_path / Path("all-downsample-64x.zarr")
split_path = base_path / Path("all-downsample-2x-split.csv")
# useful_chunk_path = base_path / Path("all-downsample-2x-masks-only.csv")  # just masks
useful_chunk_path = base_path / Path("all-downsample-2x.csv")  # all non-zero data

runs_path = Path("/mnt/efs/dlmbl/G-bs/runs/")
run_name = datetime.datetime.now().strftime("%m-%d") + "-2d-scalenet"
logger = SummaryWriter(runs_path / run_name)

transform = RandRotate(range_x=np.pi / 16, prob=0.5, padding_mode="zeros")

train_dataset, val_dataset = [
    dataloader.GutDataset(
        (dataset_path_high, dataset_path_low),
        split_path,
        data_channel_name="Phase3D",
        z_split_width=0,
        split_mode=split_mode,
        useful_chunk_path=useful_chunk_path,
        transform=transform,
        patch_size=256,
        ndim=2,
    )
    for split_mode in ["train", "test"]
]

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


unet_lowres = scalenet.UNetModule(
        in_channels = [1,0,0,0],
        num_fmaps=32, 
        depth=4,
        padding="valid"
    )
unet_highres = scalenet.UNetModule(
        in_channels = [1, 0, 0, 32],
        num_fmaps = 64,
        depth=4, 
        padding="same"
    )
model = scalenet.SimpleScaleNet(unet_highres, unet_lowres, 1, torch.nn.Sigmoid())

loss_function = (
    evaluation.DiceCoefficient()
)  # torch.nn.BCEWithLogitsLoss()  # torch.nn.BCELoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
validation_metric = evaluation.f_beta(beta=1)

for epoch in range(n_epochs):
    torch.save(
        model.state_dict(),
        f"/mnt/efs/dlmbl/G-bs/models/{run_name}_model_epoch_{epoch+1}.pth",
    )
    print(f"Model weights saved for epoch {epoch}")
    train.train_multiscale(
        model,
        train_dataloader,
        optimizer,
        epoch,
        tb_logger=logger,
        device="cuda",
        loss_function=loss_function,
    )

    torch.save(
        model.state_dict(),
        f"/mnt/efs/dlmbl/G-bs/models/{run_name}_model_epoch_{epoch+1}.pth",
    )
    print(f"Model weights saved for epoch {epoch+1}")

    evaluation.validate(
        model,
        val_dataloader,
        metric=validation_metric,
        step=epoch * len(train_dataloader),
        tb_logger=logger,
        device="cuda",
    )
