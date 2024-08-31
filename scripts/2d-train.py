import torch
import numpy as np

from iohub import open_ome_zarr
from pathlib import Path

from torch.utils.data import DataLoader
from torch.nn import Sigmoid
from monai.transforms import RandRotate
from torch.utils.tensorboard import SummaryWriter
from dl_mbl_gut import dataloader, model, train, evaluation

# Input paths
base_path = Path("/mnt/efs/dlmbl/G-bs/data/")
dataset_path = base_path / Path("all-downsample-2x.zarr")
split_path = base_path / Path("all-downsample-2x-split.csv")
useful_chunk_path = base_path / Path("all-downsample-2x-masks-only.csv")  # everything

runs_path = Path("/mnt/efs/dlmbl/G-bs/runs/")
logger = SummaryWriter(runs_path / "2d-test-weighted-bce-dice")

transform = RandRotate(range_x=np.pi / 8, prob=1.0, padding_mode="zeros")

train_dataset, val_dataset = [
    dataloader.GutDataset(
        dataset_path,
        split_path,
        split_mode=split_mode,
        useful_chunk_path=useful_chunk_path,
        transform=transform,
        patch_size=428,
        ndim=2,
    )
    for split_mode in ["train", "test"]
]

train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=10, shuffle=False)

model = model.UNet(
    depth=4,
    in_channels=1,
    out_channels=1,
    kernel_size=3,
    num_fmaps=64,
    fmap_inc_factor=2,
    padding="valid",
    final_activation=Sigmoid(),
    ndim=2,
)

validation_metric = evaluation.f_beta(beta=1)
loss = train.AddLossFuncs(train.DiceCoefficient, torch.nn.BCELoss)

n_epochs = 100
for epoch in range(n_epochs):
    train.train(
        model,
        train_dataloader,
        epoch,
        tb_logger=logger,
        device="cuda",
        loss_function= loss,
    )
    
    evaluation.validate(
        model,
        val_dataloader,
        metric=validation_metric,
        step=epoch * len(train_dataloader),
        tb_logger=logger,
        device="cuda",
    )
