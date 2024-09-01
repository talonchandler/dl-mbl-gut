import numpy as np
from iohub import open_ome_zarr
from skimage.transform import downscale_local_mean
from pathlib import Path

downsample_factors = [2, 4, 8]

for downsample_factor in downsample_factors:
    # Input paths
    base_path = Path("/mnt/efs/dlmbl/G-bs/data/")
    dataset_path = Path("all-downsample-2x.zarr")
    output_path = Path(f"all-downsample-{2*downsample_factor}x.zarr")

    input_dataset = open_ome_zarr(base_path / dataset_path, mode="r", layout="hcs")

    output_dataset = open_ome_zarr(
        base_path / output_path,
        mode="w",
        layout="hcs",
        channel_names=input_dataset.channel_names,
    )

    for fish_id, input_position in input_dataset.positions():
        print("Downsampling ", fish_id)
        row, col, fov = fish_id.split("/")
        out_position = output_dataset.create_position(row, col, fov)

        # Write metadata
        for key in input_position.zattrs.keys():
            out_position.zattrs[key] = input_position.zattrs[key]
        out_position.zattrs["downsample_factor"] = downsample_factor

        out_zyx_shape = downscale_local_mean(
            input_position[0][0, 0], (1, downsample_factor, downsample_factor)
        ).shape

        out_array = out_position.create_zeros(
            name="0",
            shape=(1, input_position[0].shape[1],) + out_zyx_shape,
            dtype=input_position[0].dtype,
            chunks=(1, 1, 1, 1024, 1024),
        )

        for i in range(input_position[0].shape[1]):
            print(f"Downsampling channel {i}")
            in_array = input_position[0][0, i]
            out_data = downscale_local_mean(
                in_array, (1, downsample_factor, downsample_factor)
            )
            out_position.zattrs[f"{input_dataset.channel_names[i]}_mean"] = np.mean(
                out_data
            )
            out_position.zattrs[f"{input_dataset.channel_names[i]}_std"] = np.std(
                out_data
            )

            out_array[0, i] = out_data
