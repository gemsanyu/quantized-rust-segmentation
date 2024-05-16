import math

import torch
from torch import nn
from pretrainedmodels.models.senet import Bottleneck, SEModule
import torch.ao.quantization
import torch.nn.quantized

class QSEModule(SEModule):
    def __init__(self, channels, reduction):
        super().__init__(channels, reduction)
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()
            
    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        module_input = self.dequant(module_input)
        x = self.dequant(x)
        y = module_input*x
        return self.quant(y)


class QBottleneck(nn.Module):
    """
    Base class for bottlenecks that implements `forward()` method.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.nnq = torch.nn.quantized.FloatFunctional()
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.nnq.add(self.se_module(out), residual)
        out = self.relu(out)

        return out
        
        
class QSEResNeXtBottleneck(QBottleneck):
    """
    ResNeXt bottleneck type C with a Squeeze-and-Excitation module.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                    downsample=None, base_width=4):
        super(QSEResNeXtBottleneck, self).__init__()
        width = math.floor(planes * (base_width / 64)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False,
                                stride=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                                padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = QSEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride