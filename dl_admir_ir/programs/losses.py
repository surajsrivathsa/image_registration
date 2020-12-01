import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.utils.data as Data
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms




class Admir_Loss(nn.Module):
    def __init__(self, return_map=False, reduction="mean", eps=1e-8):
        self.reduction = reduction
        self.eps = eps
        self._return_map = return_map


    def normalized_cross_correlation(self, x, y, return_map):
        """ N-dimensional normalized cross correlation (NCC)
        Args:
            x (~torch.Tensor): Input tensor.
            y (~torch.Tensor): Input tensor.
            return_map (bool): If True, also return the correlation map.
            reduction (str, optional): Specifies the reduction to apply to the output:
                ``'mean'`` | ``'sum'``. Defaults to ``'sum'``.
            eps (float, optional): Epsilon value for numerical stability. Defaults to 1e-8.
        Returns:
            ~torch.Tensor: Output scalar
            ~torch.Tensor: Output tensor
        """

        shape = x.shape
        b = shape[0]

        # reshape
        x = x.view(b, -1)
        y = y.view(b, -1)

        # mean
        x_mean = torch.mean(x, dim=1, keepdim=True)
        y_mean = torch.mean(y, dim=1, keepdim=True)

        # deviation
        x = x - x_mean
        y = y - y_mean

        dev_xy = torch.mul(x,y)
        dev_xx = torch.mul(x,x)
        dev_yy = torch.mul(y,y)

        dev_xx_sum = torch.sum(dev_xx, dim=1, keepdim=True)
        dev_yy_sum = torch.sum(dev_yy, dim=1, keepdim=True)

        ncc = torch.div(dev_xy + self.eps / dev_xy.shape[1],
                        torch.sqrt( torch.mul(dev_xx_sum, dev_yy_sum)) + self.eps)
        ncc_map = ncc.view(b, *shape[1:])

        # reduce
        if self.reduction == 'mean':
            ncc = torch.mean(torch.sum(ncc, dim=1))
        elif self.reduction == 'sum':
            ncc = torch.sum(ncc)
        else:
            raise KeyError('unsupported reduction type: %s' % self.reduction)

        if not return_map:
            return ncc

        return ncc, ncc_map


    def forward(self, x, y):

        return self.normalized_cross_correlation(x, y,self._return_map, self.reduction, self.eps)

