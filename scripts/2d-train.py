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
n_epochs = 400
batch_size = 40
learning_rate = 1e-6
# Input paths
base_path = Path("/mnt/efs/dlmbl/G-bs/data/")
dataset_path = base_path / Path("all-downsample-8x.zarr")
split_path = base_path / Path("all-downsample-2x-split.csv")
# useful_chunk_path = base_path / Path("all-downsample-2x-masks-only.csv")  # just masks
useful_chunk_path = base_path / Path("all-downsample-2x.csv")  # all non-zero data

runs_path = Path("/mnt/efs/dlmbl/G-bs/runs/")

load_from = "/mnt/efs/dlmbl/G-bs/models/09-02-2d-large-fov-same-lr1e-6_model_epoch_99.pth"
if load_from is None:
    exp_name = "-2d-large-fov-same-lr1e-6"
    run_name = datetime.datetime.now().strftime("%m-%d") + exp_name
    start_epoch = 0
else:
    run_name = load_from.split("/")[-1].split("_model_epoch_")[0]
    start_epoch = int(load_from.split("/")[-1].split("_model_epoch_")[1].split(".pth")[0])

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
    )
    for split_mode in ["train", "test"]
]

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = model.UNet(
    depth=4,
    in_channels=1,
    out_channels=1,
    kernel_size=3,
    num_fmaps=64,
    fmap_inc_factor=2,
    padding="same",
    final_activation=Sigmoid(),
    ndim=2,
)

loss_function = (
    evaluation.DiceCoefficient()
)  # torch.nn.BCEWithLogitsLoss()  # torch.nn.BCELoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
validation_metric = evaluation.f_beta(beta=1)

if load_from is not None:
    model.load_state_dict(torch.load(load_from))

for epoch in range(start_epoch, n_epochs,1):
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
        log_image_interval=100
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
