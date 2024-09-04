import numpy as np
import os
from aicsimageio.readers import TiffReader
import torch
from torch.utils.data import Dataset, Subset
import pandas as pd
from monai import transforms
from monai.data import MetaTensor
class NucleiDataset(Dataset):
    """A PyTorch dataset to load cell images and nuclei masks"""

    def __init__(self, root_dir=".", transform=None, img_transform=None, mask_transform = None, traintestval = None, downsample_factor = None):
        self.root_dir = root_dir  # the directory with all the training samples
        self.df = pd.read_csv(self.root_dir + 'celldata.csv') #info about all of the cells
        self.samples = self.df[self.df.trainclass == traintestval.split('_inf')[0]].cell.to_list()  # list the samples
        self.traintestval = traintestval # use 'train' or 'val' for training data or 'test', 'train_inf', or 'val_inf' for inference data
        self.downfact = downsample_factor

        self.transform = (
            transform  # transformations to apply to both inputs and targets
        )

        self.img_transform = img_transform  # transformations to apply to raw image only
        self.mask_transform = mask_transform # transformations to apply to mask image only
        #  transformations to apply just to inputs
        self.inp_transforms = transforms.Compose(
            [
                transforms.NormalizeIntensity(0, 1),
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
        img_path = os.path.join(self.root_dir, 'raw/', self.samples[idx] + '_raw.tiff')
        image = TiffReader(img_path).data[1,:,:,:]
        image = image/np.percentile(image,99.5)
        # get the crop if and crop if making test data
        if self.traintestval in ['test','train_inf','val_inf']:
            oz, oy, ox = image.shape
            (z_shape,_), (y_shape, x_shape) = np.ceil(np.array([[oz,0], [oy,ox]])/ self.downfact) * self.downfact
            zdiff, ydiff, xdiff = z_shape-oz, y_shape - oy, x_shape - ox
            image = np.pad(image, ((int(np.floor(zdiff/2)), int(np.ceil(zdiff/2))), 
                                (int(np.floor(ydiff/2)), int(np.ceil(ydiff/2))), 
                                (int(np.floor(xdiff/2)), int(np.ceil(xdiff/2)))))
            #indices to insert the cropped image later
            padd = np.array([zdiff, ydiff, xdiff])
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
            self.img_transform.set_random_state(seed)
            image = self.img_transform(image)
        if self.mask_transform is not None:
            self.mask_transform.set_random_state(seed)
            mask = self.mask_transform(mask)
        if self.traintestval in ['test','train_inf','val_inf']:
            return self.samples[idx], padd, image, mask
        else:
            return image, mask  


if __name__ == '__main__':

    datadir = '/mnt/efs/dlmbl/G-bs/AvL/'

    img_transform = transforms.Compose([
            transforms.RandSpatialCrop((56,102,102), random_size = False), #min size for AvL images is 59
            transforms.RandRotate90(prob = 0.75, spatial_axes = (1,2)),
            transforms.RandRotate(prob = 0.1, range_x = np.pi*90/180),
            transforms.CenterSpatialCrop((56,72,72)), #min size for AvL images is 59
            transforms.RandAxisFlip(prob = 0.75),
            transforms.RandScaleIntensityFixedMean(prob=1.0, factors=(0,4)),
            transforms.RandGaussianNoise(prob=0.1, mean=0.0, std=1.0)
    ])


    mask_transform = transforms.Compose([
            transforms.RandSpatialCrop((56,102,102), random_size = False), #min size for AvL images is 59
            transforms.RandRotate90(prob = 0.75, spatial_axes = (1,2)),
            transforms.RandRotate(prob = 0.1, range_x = np.pi*90/180, mode='nearest'),
            transforms.CenterSpatialCrop((56,72,72)), #min size for AvL images is 59
            transforms.RandAxisFlip(prob = 0.75),
    ])

    testdataset = NucleiDataset(root_dir=datadir,
                                img_transform = img_transform,
                                mask_transform=mask_transform,
                                traintestval = 'train')
    # crop, image, mask = testdataset[5]

    # print(crop, image.shape, mask.shape)

    # For viewing random patches
    import napari
    import random

    v = napari.Viewer()

    for i in range(100):
        random_index = random.randint(0, len(testdataset))
        data, mask = testdataset[random_index]
        v.add_image(data, name="data")
        v.add_labels(np.uint8(mask), name="mask", opacity=0.25)
        input("Press Enter to continue...")
        v.layers.clear()
