import torch
import numpy as np

from iohub import open_ome_zarr
from pathlib import Path

from torch.utils.data import DataLoader
from torch.nn import Sigmoid
from monai.transforms import RandRotate
from torch.utils.tensorboard import SummaryWriter
from dl_mbl_gut import dataloader, model, train, evaluation
import datetime

# Parameters
n_epochs = 100
batch_size = 30
learning_rate = 1e-5

# Input paths
base_path = Path("/mnt/efs/dlmbl/G-bs/data/")
dataset_path = base_path / Path("all-downsample-8x.zarr")
split_path = base_path / Path("all-downsample-2x-split.csv")
# useful_chunk_path = base_path / Path("all-downsample-2x-masks-only.csv")  # just masks
useful_chunk_path = base_path / Path("all-downsample-2x.csv")  # all non-zero data

runs_path = Path("/mnt/efs/dlmbl/G-bs/runs/")
run_name = datetime.datetime.now().strftime("%m-%d") + "-2d-sdt"
logger = SummaryWriter(runs_path / run_name)

transform = RandRotate(range_x=np.pi / 16, prob=0.5, padding_mode="zeros")

train_dataset, val_dataset = [
    dataloader.GutDataset(
        dataset_path,
        split_path,
        data_channel_name="Phase3D",
        z_split_width=0,
        split_mode=split_mode,
        useful_chunk_path=useful_chunk_path,
        transform=transform,
        patch_size=256,
        ndim=2,
        signed_distance_transform=True,
    )
    for split_mode in ["train", "test"]
]

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

model = model.UNet(
    depth=4,
    in_channels=1,
    out_channels=1,
    kernel_size=3,
    num_fmaps=64,
    fmap_inc_factor=2,
    padding="same",
    final_activation=torch.nn.Tanh(),
    ndim=2,
)


class WeightedMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, prediction, target):
        weight_mask = torch.zeros_like(target)
        weight_mask[prediction > 0] = 0.1 # custom weights for testing purposes (approximately 10x more bkg voxels than gut voxels)
        weight_mask[prediction < 0] = 0.9
        return torch.mean(weight_mask * (prediction - target) ** 2)


loss_function = WeightedMSELoss()
# torch.nn.BCEWithLogitsLoss()  # torch.nn.BCELoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
validation_metric = evaluation.DiceMetric(threshold=0)

for epoch in range(n_epochs):
    torch.save(
        model.state_dict(),
        f"/mnt/efs/dlmbl/G-bs/models/{run_name}_model_epoch_{epoch+1}.pth",
    )
    print(f"Model weights saved for epoch {epoch}")
    train.train(
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
