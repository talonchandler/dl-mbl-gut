import torch
import numpy as np
import torch.nn as nn


class DiceCoefficient(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    # the dice coefficient of two sets represented as vectors a, b can be
    # computed as (2 *|a b| / (a^2 + b^2))
    def forward(self, prediction, target):
        intersection = (prediction * target).sum()
        union = (prediction * prediction).sum() + (target * target).sum()
        return 1 - (2 * intersection / union.clamp(min=self.eps))


def center_crop(x, y):
    """Center-crop x to match spatial dimensions given by y."""

    x_target_size = x.size()[:2] + y.size()[2:]

    offset = tuple((a - b) // 2 for a, b in zip(x.size(), x_target_size))

    slices = tuple(slice(o, o + s) for o, s in zip(offset, x_target_size))

    return x[slices]


def train(
    model,
    loader,
    epoch,
    log_interval=100,
    log_image_interval=20,
    tb_logger=None,
    device=None,
    early_stop=False,
    loss_function=nn.BCELoss(),
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.004)

    if device is None:
        # You can pass in a device or we will default to using
        # the gpu. Feel free to try training on the cpu to see
        # what sort of performance difference there is
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    # set the model to train mode
    model.train()

    # move model to device
    model = model.to(device)

    # iterate over the batches of this epoch
    for batch_id, (x, y) in enumerate(loader):
        # move input and target to the active device (either cpu or gpu)
        x, y = x.to(device), y.to(device)

        # zero the gradients for this iteration
        optimizer.zero_grad()

        # apply model and calculate loss
        prediction = model(x)
        print(y.shape, prediction.shape)
        if prediction.shape != y.shape:
            y = center_crop(y, prediction)
        if y.dtype != prediction.dtype:
            y = y.type(prediction.dtype)
        loss = loss_function(prediction, y)

        print(f"Prediction min: {prediction.min()}, max: {prediction.max()}")

        # backpropagate the loss and adjust the parameters
        loss.backward()
        optimizer.step()

        # log to console
        if batch_id % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_id * len(x),
                    len(loader.dataset),
                    100.0 * batch_id / len(loader),
                    loss.item(),
                )
            )

        # log to tensorboard
        if tb_logger is not None:
            step = epoch * len(loader) + batch_id
            tb_logger.add_scalar(
                tag="train_loss", scalar_value=loss.item(), global_step=step
            )
            # check if we log images in this iteration
            if step % log_image_interval == 0:
                tb_logger.add_images(
                    tag="input", img_tensor=x.to("cpu"), global_step=step
                )
                tb_logger.add_images(
                    tag="target", img_tensor=y.to("cpu"), global_step=step
                )
                tb_logger.add_images(
                    tag="prediction",
                    img_tensor=prediction.to("cpu").detach(),
                    global_step=step,
                )

        if early_stop and batch_id > 5:
            print("Stopping test early!")
            break
