import math
import torch
from dl_mbl_gut.model import ConvBlock, Downsample, CropAndConcat, OutputConv
from typing import Literal, Optional, Tuple
import numpy as np
class CropAndConcatSameSize(torch.nn.Module):
    def crop(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Center-crop x to match spatial dimensions given by y."""
        spatial_size_1 = np.array(x.shape[2:])
        spatial_size_2 = np.array(y.shape[2:])
        target_spatial_size = torch.Size(np.min([spatial_size_1, spatial_size_2], axis=0))
        x_target_size = x.size()[:2] + target_spatial_size
        y_target_size = y.size()[:2] + target_spatial_size
        print(f"{x.size()} -> {x_target_size}")
        print(f"{y.size()} -> {y_target_size}")

        x_offset = tuple((a - b) // 2 for a, b in zip(x.size(), x_target_size))
        y_offset = tuple((a-b)//2 for a, b in zip(y.size(), y_target_size))
        

        x_slices = tuple(slice(o, o + s) for o, s in zip(x_offset, x_target_size))
        y_slices = tuple(slice(o, o+s) for o, s in zip(y_offset, y_target_size))

        return x[x_slices], y[y_slices]

    def forward(
        self, arr1: torch.Tensor, arr2: torch.Tensor
    ) -> torch.Tensor:
        arr1_cropped, arr2_cropped = self.crop(arr1, arr2)

        return torch.cat([arr1_cropped, arr2_cropped], dim=1)
    
class UNetModule(torch.nn.Module):
    def __init__(
        self,
        depth: int,
        in_channels: list[int],
        num_fmaps: int = 64,
        fmap_inc_factor: int = 2,
        downsample_factor: int = 2,
        kernel_size: int = 3,
        padding: Literal["same", "valid"] = "same",
        upsample_mode: str = "nearest",
        ndim: Literal[2, 3] = 2,
        input_levels: Optional[list[int]] = None
    ):
        """A U-Net for 2D or 3D input that expects tensors shaped like:
        ``(batch, channels, height, width)`` or ``(batch, channels, depth, height, width)``,
        respectively.

        Args:
            depth (int): The number of levels in the U-Net. 2 is the smallest that really makes
                sense for the U-Net architecture.
            in_channels (int): The number of input channels in the images.
            out_channels (int, optional): How many channels the output should have. Depends on your
                task. Defaults to 1.
            final_activation (Optional[torch.nn.Module], optional): Activation to use in final
                output block. Depends on your task. Defaults to None.
            num_fmaps (int, optional): Number of feature maps in the first layer. Defaults to 64.
            fmap_inc_factor (int, optional): Factor by which to multiply the number of feature maps
                between levels. Level ``l`` will have ``num_fmaps*fmap_inc_factor**l`` feature maps.
                Defaults to 2.
            downsample_factor (int, optional): Factor for down- and upsampling of the feature maps
                between levels. Defaults to 2.
            kernel_size (int, optional): Kernel size to use in convolutions on both sides of the
                UNet. Defaults to 3.
            padding (Literal["same", "valid"], optional): The type of padding to
                use. "same" means padding is added to preserve the input dimensions.
                "valid" means no padding is added. Defaults to "same".
            upsample_mode (str, optional): The upsampling mode to pass to ``torch.nn.Upsample``.
                Usually "nearest" or "bilinear". Defaults to "nearest".
            ndim (Literal[2, 3], optional): Number of dimensions for the UNet. Use 2 for 2D-UNet and
                3 for 3D-UNet. Defaults to 2.

        Raises:
            ValueError: If unsupported values are used for padding or ndim.
        """

        super().__init__()
        if padding not in ("valid", "same"):
            msg = f"Invalid string value for padding: {padding=}. Options are same or valid."
            raise ValueError(msg)
        if ndim not in (2, 3):
            msg = f"Invalid number of dimensions: {ndim=}. Options are 2 or 3."
            raise ValueError(msg)
        self.depth = depth
        self.in_channels = in_channels

        self.num_fmaps = num_fmaps
        self.fmap_inc_factor = fmap_inc_factor
        self.downsample_factor = downsample_factor
        self.kernel_size = kernel_size
        self.padding = padding
        self.upsample_mode = upsample_mode
        self.input_levels = input_levels

        # left convolutional passes
        self.left_convs = torch.nn.ModuleList()
        for level in range(self.depth):
            fmaps_in, fmaps_out = self.compute_fmaps_encoder(level)
            self.left_convs.append(
                ConvBlock(
                    fmaps_in, fmaps_out, self.kernel_size, self.padding, ndim=ndim
                )
            )

        # right convolutional passes
        self.right_convs = torch.nn.ModuleList()
        for level in range(self.depth - 1):
            fmaps_in, fmaps_out = self.compute_fmaps_decoder(level)
            self.right_convs.append(
                ConvBlock(
                    fmaps_in,
                    fmaps_out,
                    self.kernel_size,
                    self.padding,
                    ndim=ndim,
                )
            )

        self.downsample = Downsample(self.downsample_factor, ndim=ndim)
        self.upsample = torch.nn.Upsample(
            scale_factor=self.downsample_factor,
            mode=self.upsample_mode,
        )
        self.crop_and_concat = CropAndConcat()
        self.crop_same_size = CropAndConcatSameSize()


    def compute_fmaps_encoder(self, level: int) -> Tuple[int, int]:
        """Compute the number of input and output feature maps for
        a conv block at a given level of the UNet encoder (left side).

        Args:
        ----
            level (int): The level of the U-Net which we are computing
            the feature maps for. Level 0 is the input level, level 1 is
            the first downsampled layer, and level=depth - 1 is the bottom layer.

        Output (tuple[int, int]): The number of input and output feature maps
            of the encoder convolutional pass in the given level.
        """
        if level == 0:  # Leave out function
            fmaps_in = self.in_channels[0]
        else:
            fmaps_in = self.in_channels[level] + self.num_fmaps * self.fmap_inc_factor ** (level - 1)

        fmaps_out = self.num_fmaps * self.fmap_inc_factor**level
        return fmaps_in, fmaps_out

    def compute_fmaps_decoder(self, level: int) -> Tuple[int, int]:
        """Compute the number of input and output feature maps for a conv block
        at a given level of the UNet decoder (right side). Note:
        The bottom layer (depth - 1) is considered an "encoder" conv pass,
        so this function is only valid up to depth - 2.

        Args:
        ----
            level (int): The level of the U-Net which we are computing
            the feature maps for. Level 0 is the input level, level 1 is
            the first downsampled layer, and level=depth - 1 is the bottom layer.

        Output (tuple[int, int]): The number of input and output feature maps
            of the encoder convolutional pass in the given level.
        """
        fmaps_out = self.num_fmaps * self.fmap_inc_factor ** (
            level
        )  # Leave out function
        concat_fmaps = self.compute_fmaps_encoder(level)[
            1
        ]  # The channels that come from the skip connection
        fmaps_in = concat_fmaps + self.num_fmaps * self.fmap_inc_factor ** (level + 1)

        return fmaps_in, fmaps_out

    def forward(self, *args) -> torch.Tensor:
        # left side
        convolution_outputs = []
        layer_input = args[0]
        next_input = 1
        for i in range(self.depth - 1):
            print(i, self.in_channels[i])# leave out center of for loop
            if self.in_channels[i] != 0 and i > 0:
                layer_input = self.crop_same_size(args[next_input], layer_input)
                next_input += 1
            conv_out = self.left_convs[i](layer_input)
            convolution_outputs.append(conv_out)
            downsampled = self.downsample(conv_out)
            layer_input = downsampled

        # bottom
        if self.in_channels[-1] != 0:
            layer_input = self.crop_same_size(args[next_input], layer_input)
            next_input += 1
        conv_out = self.left_convs[-1](layer_input)
        layer_input = conv_out

        # right
        for i in range(0, self.depth - 1)[::-1]:  # leave out center of for loop
            upsampled = self.upsample(layer_input)
            concat = self.crop_and_concat(convolution_outputs[i], upsampled)
            conv_output = self.right_convs[i](concat)
            layer_input = conv_output
        
        return layer_input
    
class SimpleScaleNet(torch.nn.Module):
    def __init__(self, unet_fullres, unet_lowres, out_channels, final_activation):
        super(SimpleScaleNet, self).__init__()
        self.unet_fullres: UNetModule = unet_fullres
        self.unet_lowres: UNetModule = unet_lowres
        self.final_conv = OutputConv(self.unet_fullres.num_fmaps, out_channels, final_activation)
    def forward(self, high_res, low_res):
        low_res_rep = self.unet_lowres(low_res)
        print(low_res_rep.size())
        return self.unet_fullres(high_res, low_res_rep)
    
if __name__ == "__main__":
    unet_lowres = UNetModule(
        in_channels = [1,0,0,0],
        num_fmaps=32, 
        depth=4,
        padding="valid"
    )
    unet_highres = UNetModule(
        in_channels = [1, 0, 0, 32],
        num_fmaps = 64,
        depth=4, 
        padding="same"
    )
    scalenet = SimpleScaleNet(unet_highres, unet_lowres, 1, torch.nn.Sigmoid())
    scalenet(torch.ones(1,1, 256,256), torch.ones((1, 1, 132, 132)))