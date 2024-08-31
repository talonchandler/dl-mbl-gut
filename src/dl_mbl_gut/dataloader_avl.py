from torch.utils.data import Dataset
import os
from aicsimageio.readers import TiffReader


class NucleiDataset(Dataset):
    """A PyTorch dataset to load cell images and nuclei masks"""

    def __init__(self, pop_mean = 1, pop_std = 1, root_dir=".", transform=None, img_transform=None,):
        self.root_dir = root_dir  # the directory with all the training samples
        self.pop_mean = pop_mean # population mean of the dataset for normalization
        self.pop_std = pop_std # population mean of the dataset for normalization
        self.samples = os.listdir(self.root_dir + 'raw/')  # list the samples
        self.transform = (
            transform  # transformations to apply to both inputs and targets
        )

        self.img_transform = img_transform  # transformations to apply to raw image only
        #  transformations to apply just to inputs
        inp_transforms = transforms.Compose(
            [
                transforms.Normalize(0.5, 0.5),
                transforms.ToTensor(),
            ]
        )

        self.loaded_imgs = [None] * len(self.samples)
        self.loaded_masks = [None] * len(self.samples)
        for sample_ind in range(len(self.samples)):
            img_path = os.path.join(
                self.root_dir, 'raw/', self.samples[sample_ind]
            )
            image = TiffReader(img_path).data
            normim = image/np.max(image)
            self.loaded_imgs[sample_ind] = inp_transforms(normim.astype('float32'))
            mask_path = os.path.join(
                self.root_dir, 'seg/', self.samples[sample_ind].split('_raw')[0], "_segmented.tiff"
            )
            mask = TiffReader(mask_path).data
            self.loaded_masks[sample_ind] = transforms.ToTensor()(mask)

    # get the total number of samples
    def __len__(self):
        return len(self.samples)

    # fetch the training sample given its index
    def __getitem__(self, idx):
        # we'll be using Pillow library for reading files
        # since many torchvision transforms operate on PIL images
        image = self.loaded_imgs[idx]
        mask = self.loaded_masks[idx]
        if self.transform is not None:
            # Note: using seeds to ensure the same random transform is applied to
            # the image and mask
            seed = torch.seed()
            torch.manual_seed(seed)
            image = self.transform(image)
            torch.manual_seed(seed)
            mask = self.transform(mask)
        if self.img_transform is not None:
            image = self.img_transform(image)
        return image, mask





def get_ds_stats(
        imdir: str,
        stats: list,
):
    statdict = {}
    count = 0
    howmany = len(os.listdir(imdir))
    for i in os.listdir(imdir):
        im = TiffReader(imdir + i)
        if (len(stats)>1):
            imdata = im.data
        statdict[i] = {}
        if 'mean' in stats:
            mean = np.mean(imdata)
            statdict[i]['mean'] = mean
        if 'std' in stats:
            std = np.std(imdata)
            statdict[i]['std'] = std
        if 'shape' in stats:
            shape = im.shape
            statdict[i]['shape'] = shape
        count = count +1
        print(f'Finished_{count}/{howmany}')
    return statdict
