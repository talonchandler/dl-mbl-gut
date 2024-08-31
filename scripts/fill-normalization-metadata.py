import numpy as np
from iohub import open_ome_zarr
from pathlib import Path

# Input paths
base_path = Path("/mnt/efs/dlmbl/G-bs/data/")
dataset_path = Path("all-downsample-2x.zarr")
useful_chunk_path = base_path / dataset_path.with_suffix(".csv")  # everything

dataset = open_ome_zarr(base_path / dataset_path, mode="a", layout="hcs")
positions = [x for x in dataset.positions()]

data_channel_index = dataset.get_channel_index("BF_fluor_path")

for fish_id, position in positions:  # fish id
    data = position[0][0, data_channel_index, ::2, ::2, ::2]
    print(f"Writing to {fish_id}")
    non_zero_data = data[data > 0]
    position.zattrs["mean"] = np.mean(non_zero_data)
    position.zattrs["std"] = np.std(non_zero_data)
