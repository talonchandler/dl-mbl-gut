import torch
import numpy as np

from pathlib import Path
from iohub import open_ome_zarr
from dl_mbl_gut import model
from dl_mbl_gut.dataloader import GutDataset


def apply_model_to_yx_array(trained_model: GutDataset, input_yx_array):
    input = torch.tensor(input_yx_array[None, None]).to("cpu")
    output = trained_model(input).detach().numpy()
    return output[0, 0]


if __name__ == "__main__":
    model_path = Path(
        "/mnt/efs/dlmbl/G-bs/models/09-01-2d-large-fov-same_model_epoch_10.pth"
    )
    input_path = Path("/mnt/efs/dlmbl/G-bs/data/all-downsample-8x.zarr")
    output_path = Path("/mnt/efs/dlmbl/G-bs/data/all-downsample-8x-predictions.zarr")

    my_model = model.UNet(
        depth=4,
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        num_fmaps=64,
        fmap_inc_factor=2,
        padding="same",
        final_activation=torch.nn.Sigmoid(),
        ndim=2,
    )
    my_model.load_state_dict(torch.load(model_path), strict=False)
    my_model.to("cpu")

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
            data_crop = data[:out_shape[-2], :out_shape[-1]]
            print(f"\tProcessing z_slice {i}")
            out_array[0, 0, i] = apply_model_to_yx_array(
                my_model, data_crop
            )
