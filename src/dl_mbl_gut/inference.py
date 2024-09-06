import torch
import numpy as np

from pathlib import Path
from iohub import open_ome_zarr
from dl_mbl_gut import model, model_asym
from dl_mbl_gut.dataloader import GutDataset


def apply_model_to_yx_array(trained_model: GutDataset, input_yx_array, device="cpu"):
    input = torch.tensor(input_yx_array[None, None]).to(device)
    output = trained_model(input).detach().to("cpu").numpy()
    return output[0, 0]


def apply_model_to_zyx_array(trained_model: model_asym.UNet, input_zyx_array, device="cpu"):
    input = torch.tensor(input_zyx_array[None, None, None]).to(device)
    output = trained_model(input).detach().to("cpu").numpy()
    return output[0, 0]


if __name__ == "__main__":
    device = "cpu"
    epoch = 5000#30000
    run_name = "09-03-2d-new-annotations"

    model_path = Path(
        # "/mnt/efs/dlmbl/G-bs/models/09-01-2d-large-fov-same_model_epoch_99.pth"
        # "/mnt/efs/dlmbl/G-bs/models/09-02-2d-large-fov-same-lr1e-6_model_epoch_100.pth"
        "/mnt/efs/dlmbl/G-bs/models/"
        + run_name
        + "_model_epoch_"
        + str(epoch)
        + ".pth"
    )
    input_path = Path("/mnt/efs/dlmbl/G-bs/data/all-downsample-8x.zarr")
    # output_path = Path(
    output_path = Path(
        # "/mnt/efs/dlmbl/G-bs/data/all-downsample-8x-predictions.zarr")
        # "/mnt/efs/dlmbl/G-bs/data/all-downsample-8x-predictions-lr1e-6.zarr"
        "/mnt/efs/dlmbl/G-bs/data/predictions-"
        + run_name
        + "_"
        + str(epoch)
        + ".zarr"
    )

    if "sdt" in run_name:
        final_activation = torch.nn.Tanh()
    else:
        final_activation = torch.nn.Sigmoid()

    my_model = model.UNet(
        depth=4,
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        num_fmaps=64,
        fmap_inc_factor=2,
        padding="same",
        final_activation=final_activation,
        ndim=2,
    )
    my_model.load_state_dict(torch.load(model_path), strict=False)
    my_model.to(device)

    input_dataset = open_ome_zarr(input_path, mode="r", layout="hcs")
    output_dataset = open_ome_zarr(
        output_path, mode="w", layout="hcs", channel_names=["prediction"]
    )
    for key, pos in input_dataset.positions():
        print("Processing ", key)
        row, col, fov = key.split("/")
        out_position = output_dataset.create_position(row, col, fov)
        for k in pos.zattrs.keys():
            out_position.zattrs[k] = pos.zattrs[k]

        y_shape = int(np.floor(pos[0].shape[-2] / 8) * 8)
        out_shape = (1, 1, pos[0].shape[-3]) + (y_shape, 1024)
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
        for i in range(pos[0].shape[-3]):
            raw = pos[0][0, channel_index, i]
            data = (raw - mean) / std
            data_crop = data[: out_shape[-2], : out_shape[-1]]
            print(f"\tProcessing z_slice {i}")
            out_array[0, 0, i] = apply_model_to_yx_array(
                my_model, data_crop, device=device
            )
