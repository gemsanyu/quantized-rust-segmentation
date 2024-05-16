from typing import Optional

from segmentation_models_pytorch.decoders.fpn import FPN
from segmentation_models_pytorch.decoders.fpn.decoder import FPNBlock, SegmentationBlock, MergeBlock
import torch
import torch.nn.functional as F
from torch import nn
import torch.ao.quantization
import torch.nn.quantized

from qmodel.encoder.utils import get_quantized_encoder

class QFPNBlock(FPNBlock):
    def __init__(self, pyramid_channels, skip_channels):
        super().__init__(pyramid_channels, skip_channels)
        self.nnq = torch.nn.quantized.FloatFunctional()
    
    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        skip = self.skip_conv(skip)
        x = self.nnq.add(x, skip)
        return x
    
class QMergeBlock(MergeBlock):
    def __init__(self, policy):
        super().__init__(policy)
        self.nnq = torch.nn.quantized.FloatFunctional()

    def forward(self, x):
        if self.policy == "add":
            y = x[0]
            for i in range(1, len(x)):
                y = self.nnq.add(y,x[i])
            return y
        elif self.policy == "cat":
            return torch.cat(x, dim=1)
        else:
            raise ValueError("`merge_policy` must be one of: ['add', 'cat'], got {}".format(self.policy))



class QFPNDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        encoder_depth=5,
        pyramid_channels=256,
        segmentation_channels=128,
        dropout=0.2,
        merge_policy="add",
    ):
        super().__init__()

        self.out_channels = segmentation_channels if merge_policy == "add" else segmentation_channels * 4
        if encoder_depth < 3:
            raise ValueError("Encoder depth for FPN decoder cannot be less than 3, got {}.".format(encoder_depth))

        encoder_channels = encoder_channels[::-1]
        encoder_channels = encoder_channels[: encoder_depth + 1]

        self.p5 = nn.Conv2d(encoder_channels[0], pyramid_channels, kernel_size=1)
        self.p4 = QFPNBlock(pyramid_channels, encoder_channels[1])
        self.p3 = QFPNBlock(pyramid_channels, encoder_channels[2])
        self.p2 = QFPNBlock(pyramid_channels, encoder_channels[3])

        self.seg_blocks = nn.ModuleList(
            [
                SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=n_upsamples)
                for n_upsamples in [3, 2, 1, 0]
            ]
        )

        self.merge = QMergeBlock(merge_policy)
        self.dropout = nn.Dropout2d(p=dropout, inplace=True)
        
    def forward(self, *features):
        c2, c3, c4, c5 = features[-4:]

        p5 = self.p5(c5)
        p4 = self.p4(p5, c4)
        p3 = self.p3(p4, c3)
        p2 = self.p2(p3, c2)

        feature_pyramid = [seg_block(p) for seg_block, p in zip(self.seg_blocks, [p5, p4, p3, p2])]
        x = self.merge(feature_pyramid)
        x = self.dropout(x)

        return x

class QFPN(FPN):
    def __init__(self, encoder_name: str = "resnet34", encoder_depth: int = 5, encoder_weights: Optional[str] = "imagenet", decoder_pyramid_channels: int = 256, decoder_segmentation_channels: int = 128, decoder_merge_policy: str = "add", decoder_dropout: float = 0.2, in_channels: int = 3, classes: int = 1, activation: Optional[str] = None, upsampling: int = 4, aux_params: Optional[dict] = None):
        super().__init__(encoder_name, encoder_depth, encoder_weights, decoder_pyramid_channels, decoder_segmentation_channels, decoder_merge_policy, decoder_dropout, in_channels, classes, activation, upsampling, aux_params)
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()
        
        self.encoder = get_quantized_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )
        
        self.decoder = QFPNDecoder(
            encoder_channels=self.encoder.out_channels,
            encoder_depth=encoder_depth,
            pyramid_channels=decoder_pyramid_channels,
            segmentation_channels=decoder_segmentation_channels,
            dropout=decoder_dropout,
            merge_policy=decoder_merge_policy,
        )
        
        
        
    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        self.check_input_shape(x)
        x = self.quant(x)
        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels
        masks = self.dequant(masks)
        return masks