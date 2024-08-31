from iohub import open_ome_zarr
from pathlib import Path

# Input paths
base_path = Path("/mnt/efs/dlmbl/G-bs/data/")
dataset_path = Path("all-downsample-2x.zarr")
output_path = Path("all-downsample-2x-rechunked.zarr")

input_dataset = open_ome_zarr(base_path / dataset_path, mode="r", layout="hcs")

output_dataset = open_ome_zarr(
    base_path / output_path,
    mode="w",
    layout="hcs",
    channel_names=input_dataset.channel_names,
)

for fish_id, input_position in input_dataset.positions():
    print("Rechunking ", fish_id)
    row, col, fov = fish_id.split("/")
    out_position = output_dataset.create_position(row, col, fov)
    for key in input_position.zattrs.keys():
        out_position.zattrs[key] = input_position.zattrs[key]
    out_array = out_position.create_zeros(
        name="0",
        shape=input_position[0].shape,
        dtype=input_position[0].dtype,
        chunks=(1, 1, 1, 1024, 1024),
    )

    for i in range(input_position[0].shape[1]):
        print(f"\tCopying channel {i}")
        out_array[0, i] = input_position[0][0, i]
