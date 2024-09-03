import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from dl_mbl_gut import evaluation
from dl_mbl_gut.utils import center_crop



def train(
    model,
    loader,
    optimizer,
    epoch,
    log_interval=100,
    log_image_interval=20,
    tb_logger=None,
    device=None,
    early_stop=False,
    loss_function=nn.BCELoss(),
):
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
    loss = 1
    pbar = tqdm(enumerate(loader))
    for batch_id, (x, y) in pbar:

        # move input and target to the active device (either cpu or gpu)
        x, y = x.to(device), y.to(device)

        # zero the gradients for this iteration
        optimizer.zero_grad()

        # apply model and calculate loss
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            prediction = model(x)
            print(y.shape, prediction.shape)
            if prediction.shape != y.shape:
                y = center_crop(y, prediction)
            if y.dtype != prediction.dtype:
                y = y.type(prediction.dtype)
            loss = loss_function(prediction, y)

        # backpropagate the loss and adjust the parameters
        loss.backward()
        optimizer.step()

        # Progress bar logging
        pbar.set_description(
            f"Loss: {loss:.2f}, Pred min: {prediction.min():.2f}, Pred max: {prediction.max():.2f}"
        )

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
            if (step % log_image_interval == 0) and (len(x.shape) <= 4):
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
            elif (step % log_image_interval == 0) and (len(x.shape) > 4):
                tb_logger.add_images(
                    tag="input",
                    img_tensor=np.max(x.to("cpu").numpy(), axis=-3),
                    global_step=step,
                )
                tb_logger.add_images(
                    tag="target",
                    img_tensor=np.max(y.to("cpu").numpy(), axis=-3),
                    global_step=step,
                )
                tb_logger.add_images(
                    tag="prediction",
                    img_tensor=np.max(prediction.to("cpu").detach().numpy(), axis=-3),
                    global_step=step,
                )
        if early_stop and batch_id > 5:
            print("Stopping test early!")
            break



def train_eval(
    model,
    loader,
    val_loader,
    optimizer,
    validation_metric,
    epoch,
    log_interval=100,
    log_image_interval=20,
    tb_logger=None,
    device=None,
    early_stop=False,
    loss_function=nn.BCELoss(),
    val_with_log = False,
    thresh_scan = None,
):

    if device is None:
        # You can pass in a device or we will default to using
        # the gpu. Feel free to try training on the cpu to see
        # what sort of performance difference there is
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    if not thresh_scan:
        thresh_scan = 0.5

    # set the model to train mode
    model.train()

    # move model to device
    model = model.to(device)

    # iterate over the batches of this epoch
    loss = 1
    pbar = tqdm(enumerate(loader))
    for batch_id, (x, y) in pbar:

        # move input and target to the active device (either cpu or gpu)
        x, y = x.to(device), y.to(device)

        # zero the gradients for this iteration
        optimizer.zero_grad()

        # apply model and calculate loss
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            prediction = model(x)
            print(y.shape, prediction.shape)
            if prediction.shape != y.shape:
                y = center_crop(y, prediction)
            if y.dtype != prediction.dtype:
                y = y.type(prediction.dtype)
            loss = loss_function(prediction, y)

        # backpropagate the loss and adjust the parameters
        loss.backward()
        optimizer.step()

        # Progress bar logging
        pbar.set_description(
            f"Loss: {loss:.2f}, Pred min: {prediction.min():.2f}, Pred max: {prediction.max():.2f}"
        )

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
            if (step % log_image_interval == 0) and (len(x.shape) <= 4):
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
            elif (step % log_image_interval == 0) and (len(x.shape) > 4):
                tb_logger.add_images(
                    tag="input",
                    img_tensor=np.max(x.to("cpu").numpy(), axis=-3),
                    global_step=step,
                )
                tb_logger.add_images(
                    tag="target",
                    img_tensor=np.max(y.to("cpu").numpy(), axis=-3),
                    global_step=step,
                )
                tb_logger.add_images(
                    tag="prediction",
                    img_tensor=np.max(prediction.to("cpu").detach().numpy(), axis=-3),
                    global_step=step,
                )
        if early_stop and batch_id > 5:
            print("Stopping test early!")
            break
        
        if not val_with_log:
            if batch_id == len(loader) -1:
                evaluation.validate(
                    model,
                    val_loader,
                    validation_metric,
                    step= epoch * len(loader) + batch_id,
                    tb_logger=tb_logger,
                    device=device,
                    scan=thresh_scan,
                )
        else:
            if (batch_id % log_interval == 0) and (batch_id != 0):
                evaluation.validate(
                    model,
                    val_loader,
                    validation_metric,
                    step= epoch * len(loader) + batch_id,
                    tb_logger=tb_logger,
                    device=device,
                    scan=thresh_scan,
                )



