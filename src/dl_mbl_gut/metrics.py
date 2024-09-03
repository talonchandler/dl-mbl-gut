from torch import nn


class AddLossFuncs(nn.Module):
    def __init__(self, loss1, loss2):
        super().__init__()
        self.loss1 = loss1()
        self.loss2 = loss2()

    def forward(self, prediction, target):
        val1 = self.loss1(prediction, target)
        val2 = self.loss2(prediction, target)
        return val1 + val2


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