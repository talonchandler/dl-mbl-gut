import torch
import numpy as np
import torch.nn as nn
from dl_mbl_gut.train import DiceCoefficient, center_crop


class f_beta(nn.Module):
    def __init__(self, beta: float, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.beta = beta

    def forward(self, gt, pred):
        tp = np.sum((gt * pred))
        fp = np.sum(pred) - tp
        fn = np.sum(gt) - tp
        beta_term = 1 + np.power(self.beta, 2)
        numerator = beta_term * tp
        denominator = beta_term * tp + np.power(self.beta, 2) * fp + fn
        return numerator / denominator


def validate(
    model,
    loader,
    metric,
    step=None,
    tb_logger=None,
    device=None,
):
    loss_function = DiceCoefficient()

    if device is None:
        # You can pass in a device or we will default to using
        # the gpu. Feel free to try training on the cpu to see
        # what sort of performance difference there is
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    # set model to eval mode
    model.eval()
    model.to(device)

    # running loss and metric values
    val_loss = 0
    val_metric = 0

    # disable gradients during validation
    with torch.no_grad():
        # iterate over validation loader and update loss and metric values
        for x, y in loader:
            print(x.shape, y.shape)
            x, y = x.to(device), y.to(device)
            prediction = model(x)
            if y.dtype != prediction.dtype:
                y = y.type(prediction.dtype)
            if prediction.shape != y.shape:
                y = center_crop(y, prediction)
            val_loss += loss_function(prediction, y).item()
            val_metric += metric(prediction > 0.5, y).item()

    # normalize loss and metric
    val_loss /= len(loader)
    val_metric /= len(loader)
    print('almost')
    if (tb_logger is not None) and (len(x.shape)<=4):
        print('bad')
        assert (
            step is not None
        ), "Need to know the current step to log validation results"
        tb_logger.add_scalar(tag="val_loss", scalar_value=val_loss, global_step=step)
        tb_logger.add_scalar(
            tag="val_metric", scalar_value=val_metric, global_step=step
        )
        # we always log the last validation images
        tb_logger.add_images(tag="val_input", img_tensor=x.to("cpu"), global_step=step)
        tb_logger.add_images(tag="val_target", img_tensor=y.to("cpu"), global_step=step)
        tb_logger.add_images(
            tag="val_prediction", img_tensor=prediction.to("cpu"), global_step=step
        )
    elif (tb_logger is not None) and (len(x.shape)>4):
        assert (
            step is not None
        ), "Need to know the current step to log validation results"
        tb_logger.add_scalar(tag="val_loss", scalar_value=val_loss, global_step=step)
        tb_logger.add_scalar(
            tag="val_metric", scalar_value=val_metric, global_step=step
        )
        # we always log the last validation images
        print('hello')
        tb_logger.add_images(tag="val_input", img_tensor=np.max(x.to("cpu").numpy(),axis=-3), global_step=step)
        tb_logger.add_images(tag="val_target", img_tensor=np.max(y.to("cpu").numpy(),axis=-3), global_step=step)
        tb_logger.add_images(
            tag="val_prediction", img_tensor=np.max(prediction.to("cpu").numpy(),axis=-3), global_step=step
        )

    print(
        "\nValidate: Average loss: {:.4f}, Average Metric: {:.4f}\n".format(
            val_loss, val_metric
        )
    )
