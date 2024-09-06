from pathlib import Path
from torch import nn
from dl_mbl_gut import model_asym, inference
from iohub import open_ome_zarr
import torch
import numpy as np
import pandas as pd

device = "cpu"
run_name = "09-03-3d-unet_model_epoch_861"

model_path = Path(
    # "/mnt/efs/dlmbl/G-bs/models/09-01-2d-large-fov-same_model_epoch_99.pth"
    # "/mnt/efs/dlmbl/G-bs/models/09-02-2d-large-fov-same-lr1e-6_model_epoch_100.pth"
    "/mnt/efs/dlmbl/G-bs/models/"
    + run_name
    + ".pth"
)
input_path = Path("/mnt/efs/dlmbl/G-bs/data/all-downsample-8x.zarr")
input_split = pd.read_csv("/mnt/efs/dlmbl/G-bs/data/all-downsample-2x-split.csv")
output_path = Path("/mnt/efs/dlmbl/G-bs/data/predictions-" + run_name + ".zarr")

# assemble model
num_fmaps = 64

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

allmodel.load_state_dict(torch.load(model_path), strict=False)
allmodel.to(device)

input_dataset = open_ome_zarr(input_path, mode="r", layout="hcs")
output_dataset = open_ome_zarr(
    output_path, mode="r+", layout="hcs", channel_names=["prediction"]
)

for key, pos in input_dataset.positions():
    name_list = [key for (key, _) in output_dataset.positions()]
    if key in name_list:
        continue
    print("Processing ", key)
    row, col, fov = key.split("/")
    out_position = output_dataset.create_position(row, col, fov)
    for k in pos.zattrs.keys():
        out_position.zattrs[k] = pos.zattrs[k]

    best_slice = input_split[input_split["position"] == key]["best-slice"].values[0]

    y_shape = int(np.floor(pos[0].shape[-2] / 8) * 8)
    out_shape = (1, 1, 16, y_shape, 512)
    out_array = out_position.create_zeros(
        name="0",
        shape=out_shape,
        dtype=pos[0].dtype,
        chunks=(1, 1, 1, 1024, 1024),
    )

    input_channel_name = "Phase3D"
    channel_index = input_dataset.channel_names.index(input_channel_name)
    mean = pos.zattrs[input_channel_name + "_mean"]
    std = pos.zattrs[input_channel_name + "_std"]

    raw = pos[0][0, channel_index, best_slice - 24 : best_slice + 24 : 3]
    data = (raw - mean) / std
    data_crop = data[:, : out_shape[-2], : out_shape[-1]]
    print(f"\tProcessing position {key}")

    out_array[0, 0] = inference.apply_model_to_yx_array(
        allmodel, data_crop, device=device
    )
