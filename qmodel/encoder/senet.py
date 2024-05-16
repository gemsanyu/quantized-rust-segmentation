from pretrainedmodels.models.senet import SEModule

import segmentation_models_pytorch as smp
import torch
import torch.ao.quantization

class QSEModule(SEModule):
    def __init__(self, channels, reduction):
        super().__init__(channels, reduction)
    