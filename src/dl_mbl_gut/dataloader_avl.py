import numpy as np
import os
from aicsimageio.readers import TiffReader
import torch
from torch.utils.data import Dataset
import pandas as pd
from monai import transforms
from monai.data import MetaTensor
class NucleiDataset(Dataset):
    """A PyTorch dataset to load cell images and nuclei masks"""

    def __init__(self, root_dir=".", transform=None, img_transform=None, traintestval = None,):
        self.root_dir = root_dir  # the directory with all the training samples
        self.df = pd.read_csv(self.root_dir + 'celldata.csv') #info about all of the cells
        self.samples = self.df[self.df.trainclass == traintestval].cell.to_list()  # list the samples
        
        self.transform = (
            transform  # transformations to apply to both inputs and targets
        )

        self.img_transform = img_transform  # transformations to apply to raw image only
        #  transformations to apply just to inputs
        self.inp_transforms = transforms.Compose(
            [
                transforms.NormalizeIntensity(0.5, 0.5),
                transforms.ToTensor(),
            ]
        )


    # get the total number of samples
    def __len__(self):
        return len(self.samples)

    # fetch the training sample given its index
    def __getitem__(self, idx):
        # we'll be using Pillow library for reading files
        # since many torchvision transforms operate on PIL images
        img_path = os.path.join(
        self.root_dir, 'raw/', self.samples[idx] + '_raw.tiff')
        image = TiffReader(img_path).data[1,:,:,:]
        image = image/np.max(image)
        image = self.inp_transforms(image[np.newaxis,...])
        mask_path = os.path.join(
            self.root_dir, 'seg/', self.samples[idx] + '_segmented.tiff')
        mask = TiffReader(mask_path).data[0,:,:,:]
        mask[mask>0] = 1
        mask = transforms.ToTensor()(mask[np.newaxis,...].astype(np.float32))
        seed = np.random.randint(5000)
        if self.transform is not None:
            # Note: using seeds to ensure the same random transform is applied to
            # the image and mask
            self.transform.set_random_state(seed)
            image = self.transform(image)
            self.transform.set_random_state(seed)
            mask = self.transform(mask)
        if self.img_transform is not None:
            image = self.img_transform(image)
        return image, mask


if __name__ == '__main__':

    datadir = '/mnt/efs/dlmbl/G-bs/AvL/'

    transform = transforms.Compose([
            transforms.RandRotate90(prob = 0.33, spatial_axes = (1,2)),
            transforms.RandSpatialCrop((150,150,150), random_size = False),
            transforms.RandRotate(prob = 0.1),
            transforms.RandAxisFlip(prob = 0.75),
    ])

    testdataset = NucleiDataset(root_dir=datadir, transform = transform, traintestval = 'test')
    image, mask = testdataset[0]

    print('you did it', image.dtype, mask.dtype, image.shape, mask.shape)

    # # For viewing random patches
    # import napari
    # import random

    # v = napari.Viewer()

    # for i in range(100):
    #     random_index = random.randint(0, len(testdataset))
    #     data, mask = testdataset[random_index]
    #     v.add_image(data, name="data")
    #     v.add_labels(np.uint8(mask), name="mask", opacity=0.25)
    #     input("Press Enter to continue...")
    #     v.layers.clear()
