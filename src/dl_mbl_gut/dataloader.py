import numpy as np
import torch

from pathlib import Path
from iohub import open_ome_zarr
from torch.utils.data import Dataset
from tqdm import tqdm

from monai.transforms import RandRotate, RandCropByPosNegLabel


class GutDataset(Dataset):

    def __init__(
        self,
        root_dir,
        data_channel_name="BF_fluor_path",
        mask_channel_name="label",
        z_split_width=1,
        x_split_width=1024,
        transform=None,
        img_transform=None,
        useful_chunk_path=None,
    ):
        self.root_dir = root_dir
        self.data_channel_name = data_channel_name
        self.mask_channel_name = mask_channel_name
        self.dataset = open_ome_zarr(root_dir)
        self.positions = [x for x in self.dataset.positions()]
        self.x_split_width = x_split_width
        self.z_split_width = z_split_width
        self.transform = transform
        self.img_transform = img_transform

        self.data_channel_index = self.dataset.get_channel_index(data_channel_name)
        self.mask_channel_index = self.dataset.get_channel_index(mask_channel_name)

        # Split into x_split_width sized chunks
        self.item_keys = []

        # If useful_chunk_table is not None, load the indices from the file
        if useful_chunk_path is not None:
            with open(useful_chunk_path, "r") as f:
                self.item_keys = [
                    list(line.strip().split(",")) for line in f.readlines()
                ]
            for row in self.item_keys:
                row[1:] = [int(val) for val in row[1:]]
        else:
            for fish_id, position in self.positions:  # fish id
                z_width = position[0].shape[-3]
                x_width = position[0].shape[-1]
                for z_index in range(z_width):
                    for x_start in range(0, x_width, self.x_split_width):
                        x_end = min(x_start + self.x_split_width, x_width)
                        self.item_keys.append((fish_id, z_index, x_start, x_end))

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

        data = self.dataset[position_key][0][
            0,
            self.data_channel_index,
            z_index : (z_index + self.z_split_width),
            :,
            x_start:x_end,
        ][None]
        mask = self.dataset[position_key][0][
            0,
            self.mask_channel_index,
            z_index : (z_index + self.z_split_width),
            :,
            x_start:x_end,
        ][None]
        seed = np.random.randint(0, 100)

        if self.transform is not None:
            self.transform.set_random_state(seed)
            data = self.transform(data)
            self.transform.set_random_state(seed)
            mask = self.transform(mask)
        if self.img_transform is not None:
            data = self.img_transform(data)
            mask = self.img_transform

        crop = RandCropByPosNegLabel((1, 512, 512), None, pos=0.8, neg=0.2)
        crop.set_random_state(seed)
        data = crop(data, label=mask)
        crop.set_random_state(seed)
        mask = crop(mask, label=mask)

        return data, mask

    def info(self):
        print(f"Found {len(self)} items.")
        for position in self.dataset.positions():
            key, pos = position
            print(f"Position {key}\t has shape \t{pos[0].shape}.")


if __name__ == "__main__":
    base_path = Path("/mnt/efs/dlmbl/G-bs/data/")
    dataset_path = Path("all-downsample-2x.zarr")
    useful_chunk_path = base_path / dataset_path.with_suffix(".csv")
    # useful_chunk_path = base_path / "all-downsample-2x-masks-only.csv"

    transform = RandRotate(range_x=np.pi / 8, prob=1.0, padding_mode="zeros")

    dataset = GutDataset(
        base_path / dataset_path,
        useful_chunk_path=useful_chunk_path,
        transform=transform,
    )
    dataset.info()

    # For finding useful chunks
    # dataset._find_useful_chunks(useful_chunk_path) # this will take a while, call once to save keys

    # # For viewing random patches
    # import napari
    # import random

    # v = napari.Viewer()

    # for i in range(100):
    #     random_index = random.randint(0, len(dataset))
    #     data, mask = dataset[random_index]
    #     v.add_image(data, name="data")
    #     v.add_labels(np.uint8(mask), name="mask", opacity=0.25)
    #     input("Press Enter to continue...")
    #     v.layers.clear()
