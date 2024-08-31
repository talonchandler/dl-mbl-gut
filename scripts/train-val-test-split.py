import numpy as np
from iohub import open_ome_zarr
from pathlib import Path

# Input paths
base_path = Path("/mnt/efs/dlmbl/G-bs/data/")
dataset_path = Path("all-downsample-2x.zarr")
output_path = base_path / dataset_path.with_stem(
    dataset_path.stem + "-split"
).with_suffix(".csv")

dataset = open_ome_zarr(base_path / dataset_path, mode="a", layout="hcs")
position_names = [name for (name, pos) in dataset.positions()]


# Assign labels to positions
labels = np.random.choice(
    ["train", "val", "test"], size=len(position_names), p=[0.75, 0.15, 0.1]
)

# Create a list of (position, label) tuples
position_labels = list(zip(position_names, labels))

# Save position labels to output file
with open(output_path, "w") as f:
    f.write("position,label\n")
    for position, label in position_labels:
        f.write(f"{position},{label}\n")
