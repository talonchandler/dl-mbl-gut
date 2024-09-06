import numpy as np
import torch

from pathlib import Path
from iohub import open_ome_zarr
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Literal

from monai.transforms import (
    RandRotate,
    RandCropByPosNegLabel,
    RandWeightedCrop,
    NormalizeIntensity,
    CenterSpatialCrop,
    RandGaussianSharpen,
    Compose
)
import time
from scipy.ndimage import distance_transform_edt, map_coordinates

def compute_sdt(labels: np.ndarray, scale: int = 5):
    """Function to compute a signed distance transform."""
    dims = len(labels.shape)
    # Create a placeholder array of infinite distances
    distances = np.ones(labels.shape, dtype=np.float32) * np.inf
    for axis in range(dims):
        # Here we compute the boundaries by shifting the labels and comparing to the original labels
        # This can be visualized in 1D as:
        # a a a b b c c c
        #   a a a b b c c c
        #   1 1 0 1 0 1 1
        # Applying a half pixel shift makes the result more obvious:
        # a a a b b c c c
        #  1 1 0 1 0 1 1
        bounds = (
            labels[*[slice(None) if a != axis else slice(1, None) for a in range(dims)]]
            == labels[
                *[slice(None) if a != axis else slice(None, -1) for a in range(dims)]
            ]
        )
        # pad to account for the lost pixel
        bounds = np.pad(
            bounds,
            [(1, 1) if a == axis else (0, 0) for a in range(dims)],
            mode="constant",
            constant_values=1,
        )
        # compute distances on the boundary mask
        axis_distances = distance_transform_edt(bounds)

        # compute the coordinates of each original pixel relative to the boundary mask and distance transform.
        # Its just a half pixel shift in the axis we computed boundaries for.
        coordinates = np.meshgrid(
            *[
                range(axis_distances.shape[a])
                if a != axis
                else np.linspace(0.5, axis_distances.shape[a] - 1.5, labels.shape[a])
                for a in range(dims)
            ],
            indexing="ij",
        )
        coordinates = np.stack(coordinates)

        # Interpolate the distances to the original pixel coordinates
        sampled = map_coordinates(
            axis_distances,
            coordinates=coordinates,
            order=3,
        )

        # Update the distances with the minimum distance to a boundary in this axis
        distances = np.minimum(distances, sampled)

    # Normalize the distances to be between -1 and 1
    distances = np.tanh(distances / scale)

    # Invert the distances for pixels in the background
    distances[labels == 0] *= -1
    return distances


class GutDataset(Dataset):

    def __init__(
        self,
        root_dir: Path,
        split_path: Path,
        split_mode: Literal["train", "val", "test", None] = "train",
        data_channel_name: str = "BF_fluor_path",
        mask_channel_name: str = "label",
        z_split_width: int = 0,
        z_stride: int = 1,
        x_split_width: int = 1024,
        patch_size: int = 256,
        transform=None,
        img_transform=None,
        useful_chunk_path=None,
        ndim=3,
        signed_distance_transform=False,
        new_annotations=False,
        pos_frac=0.5,
    ):
        self.root_dir = root_dir
        self.split_path = split_path
        self.split_mode = split_mode
        self.data_channel_name = data_channel_name
        self.mask_channel_name = mask_channel_name
        self.x_split_width = x_split_width
        self.z_split_width = z_split_width
        self.z_stride = z_stride
        self.patch_size = patch_size
        self.transform = transform
        self.img_transform = img_transform
        self.ndim = ndim
        self.signed_distance_transform = signed_distance_transform
        self.new_annotations = new_annotations
        self.pos_frac = pos_frac

        self.dataset = open_ome_zarr(self.root_dir)
        self.fish_ids = []
        self.in_focus_slices = []
        with open(self.split_path, "r") as f:
            for line in f.readlines():
                row = line.strip().split(",")
                if row[1] == self.split_mode:
                    self.fish_ids.append(row[0])
                    self.in_focus_slices.append(int(row[-1]))

        self.data_channel_index = self.dataset.get_channel_index(data_channel_name)
        self.mask_channel_index = self.dataset.get_channel_index(mask_channel_name)

        # Split into x_split_width sized chunks
        self.item_keys = []

        # If useful_chunk_table is not None, load the indices from the file
        if useful_chunk_path is not None:
            with open(useful_chunk_path, "r") as f:
                for line in f.readlines():
                    row = line.strip().split(",")
                    for i, fish_id in enumerate(self.fish_ids):
                        if row[0] == fish_id: # only use if in split
                            self.item_keys.append(row)

            # Clean types
            for row in self.item_keys:
                row[1:] = [int(val) for val in row[1:]]

        else:
            for fish_id, position in self.dataset.positions():  # fish id
                z_width = position[0].shape[-3]
                x_width = position[0].shape[-1]
                for z_index in range(z_width):
                    for x_start in range(0, x_width, self.x_split_width):
                        x_end = min(x_start + self.x_split_width, x_width)
                        self.item_keys.append((fish_id, z_index, x_start, x_end))

        # Overwrite item keys with new annotations only 
        if self.new_annotations: 
            self.item_keys = []
            for i, fish_id in enumerate(self.fish_ids):
                row = (fish_id, self.in_focus_slices[i], 0, 1024)
                self.item_keys.append(row)
                print(row)



    def _find_useful_chunks(self, output_path: Path):
        """Read in all data and save the indices of chunks that contain a mask."""
        useful_item_keys = []
        for i in tqdm(range(len(self))):
            data, mask = self[i]
            if np.max(data) > 0:  # and np.max(mask) > 0:
                print("Found useful chunk.")
                useful_item_keys.append(self.item_keys[i])

        print(f"Found {len(useful_item_keys)} useful chunks.")
        with open(output_path, "w") as f:
            for item_key in useful_item_keys:
                f.write(f"{item_key[0]},{item_key[1]},{item_key[2]},{item_key[3]}\n")

    def __len__(self):
        return len(self.item_keys)

    def __getitem__(self, idx: int):
        position_key, z_index, x_start, x_end = self.item_keys[idx]

        # Always get a valid z range
        z_shape = self.dataset[position_key][0].shape[-3]
        z_width = 2 * self.z_split_width + 1
        z_min = z_index - self.z_split_width
        z_max = z_index + self.z_split_width + 1
        if z_min <= 0:
            z_min = 0
            z_max = 2 * self.z_split_width + 1
        if z_max >= z_shape:
            z_min = z_shape - 2 * self.z_split_width - 1 - 2  # kludge
            z_max = z_shape - 2

        data = self.dataset[position_key][0][
            0,
            self.data_channel_index,
            z_min:z_max:self.z_stride,
            :,
            :,
        ][None]
        mask = self.dataset[position_key][0][
            0,
            self.mask_channel_index,
            z_min:z_max:self.z_stride,
            :,
            :,
        ][None]

        z_out_shape = mask.shape[-3]
        print(mask.shape, data.shape)

        if self.signed_distance_transform:
            interim_mask = compute_sdt(mask[0, 0], scale=10)[None, None]
        else:
            interim_mask = mask

        seed = np.random.randint(0, 100)

        outer_patch_size = np.ceil(self.patch_size * np.sqrt(2)).astype(int)
        y_crop_width = np.min([outer_patch_size, mask.shape[-2]])
        crop = RandCropByPosNegLabel(
            (z_out_shape, y_crop_width, outer_patch_size),
            None,
            pos=self.pos_frac,
            neg=1 - self.pos_frac,
        )

        crop.set_random_state(seed)
        data = crop(data, label=mask)[0]
        crop.set_random_state(seed)
        mask = crop(interim_mask, label=mask)[0]

        if self.transform is not None:
            self.transform.set_random_state(seed)
            data = self.transform(data)
            self.transform.set_random_state(seed)
            mask = self.transform(mask)
        if self.img_transform is not None:
            data = self.img_transform(data)
            mask = self.img_transform

        # Crop inner path
        center_crop = CenterSpatialCrop((z_out_shape, self.patch_size, self.patch_size))
        data = center_crop(data)
        mask = center_crop(mask)

        # Normalize the data
        mean = self.dataset[position_key].zattrs[self.data_channel_name + "_mean"]
        std = self.dataset[position_key].zattrs[self.data_channel_name + "_std"]

        data = NormalizeIntensity(subtrahend=mean, divisor=std)(data[0])
        mask = mask[0]

        if not self.signed_distance_transform:
            mask = mask > 0

        if self.z_stride > 0:
            data = torch.unsqueeze(data, dim=0)
            mask = torch.unsqueeze(mask, dim=0)

        return data, mask

    def info(self):
        print(f"Found {len(self)} items.")
        for position in self.dataset.positions():
            key, pos = position
            print(f"Position {key}\t has shape \t{pos[0].shape}.")


if __name__ == "__main__":
    base_path = Path("/mnt/efs/dlmbl/G-bs/data/")
    #dataset_path = Path("all-downsample-8x.zarr")
    dataset_path = Path("all-downsample-8x-fix-annotation.zarr")
    # useful_chunk_path = base_path / Path("all-downsample-2x-masks-only.csv")
    useful_chunk_path = base_path / "all-downsample-2x-masks-only.csv"

    split_path = base_path / Path("all-downsample-2x-split.csv")

    transform = Compose([RandRotate(range_x=np.pi / 16, prob=0.5, padding_mode="zeros"), 
                         RandGaussianSharpen(sigma1_x=(0, 3), sigma1_y=(0, 3), sigma1_z=(0, 0), sigma2_x=(0, 0), sigma2_y=(0, 0), sigma2_z=(0, 0), prob=0.5)])

    sdt = True
    dataset = GutDataset(
        base_path / dataset_path,
        split_path=split_path,
        split_mode="train",
        data_channel_name="BF_fluor_path", #"Phase3D",
        z_split_width=0,#23,
        z_stride=1,#3,
        useful_chunk_path=useful_chunk_path,
        patch_size=256,
        transform=transform,
        signed_distance_transform=sdt,
        new_annotations=False,
    )
    dataset.info()

    from torch.utils.data import DataLoader

    batch_size = 50
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=True)

    # total_time = 0
    # num_pairs = 2
    # start_time = time.time()
    # for i, (x, y) in enumerate(dataloader):
    #     if i == num_pairs:
    #         break

    # end_time = time.time()
    # total_time += end_time - start_time

    # average_load_time = total_time / (batch_size * num_pairs)
    # print(f"Average load time per pair: {average_load_time:.2f} seconds")

    # For finding useful chunks
    # dataset._find_useful_chunks(useful_chunk_path) # this will take a while, call once to save keys

    # For viewing random patches
    import napari
    import random

    v = napari.Viewer()

    while True:
        data, mask = next(iter(dataloader))
        print(data.shape, mask.shape)
        v.add_image(data, name="data")
        if sdt:
            v.add_image(mask, name="mask", opacity=0.25, blending="additive")
        else:
            v.add_labels(np.uint8(mask), name="mask", opacity=0.25)
        input("Press Enter to continue...")
        v.layers.clear()
