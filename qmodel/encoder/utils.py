import functools

import torch.utils.model_zoo as model_zoo
from segmentation_models_pytorch.encoders.senet import SENetEncoder, pretrained_settings
import timm

from qmodel.encoder.senet_utils import QSEResNeXtBottleneck



senet_encoders = {
    # "senet154": {
    #     "encoder": SENetEncoder,
    #     "pretrained_settings": pretrained_settings["senet154"],
    #     "params": {
    #         "out_channels": (3, 128, 256, 512, 1024, 2048),
    #         "block": SEBottleneck,
    #         "dropout_p": 0.2,
    #         "groups": 64,
    #         "layers": [3, 8, 36, 3],
    #         "num_classes": 1000,
    #         "reduction": 16,
    #     },
    # },
    # "se_resnet50": {
    #     "encoder": SENetEncoder,
    #     "pretrained_settings": pretrained_settings["se_resnet50"],
    #     "params": {
    #         "out_channels": (3, 64, 256, 512, 1024, 2048),
    #         "block": SEResNetBottleneck,
    #         "layers": [3, 4, 6, 3],
    #         "downsample_kernel_size": 1,
    #         "downsample_padding": 0,
    #         "dropout_p": None,
    #         "groups": 1,
    #         "inplanes": 64,
    #         "input_3x3": False,
    #         "num_classes": 1000,
    #         "reduction": 16,
    #     },
    # },
    # "se_resnet101": {
    #     "encoder": SENetEncoder,
    #     "pretrained_settings": pretrained_settings["se_resnet101"],
    #     "params": {
    #         "out_channels": (3, 64, 256, 512, 1024, 2048),
    #         "block": SEResNetBottleneck,
    #         "layers": [3, 4, 23, 3],
    #         "downsample_kernel_size": 1,
    #         "downsample_padding": 0,
    #         "dropout_p": None,
    #         "groups": 1,
    #         "inplanes": 64,
    #         "input_3x3": False,
    #         "num_classes": 1000,
    #         "reduction": 16,
    #     },
    # },
    # "se_resnet152": {
    #     "encoder": SENetEncoder,
    #     "pretrained_settings": pretrained_settings["se_resnet152"],
    #     "params": {
    #         "out_channels": (3, 64, 256, 512, 1024, 2048),
    #         "block": SEResNetBottleneck,
    #         "layers": [3, 8, 36, 3],
    #         "downsample_kernel_size": 1,
    #         "downsample_padding": 0,
    #         "dropout_p": None,
    #         "groups": 1,
    #         "inplanes": 64,
    #         "input_3x3": False,
    #         "num_classes": 1000,
    #         "reduction": 16,
    #     },
    # },
    # "se_resnext50_32x4d": {
    #     "encoder": SENetEncoder,
    #     "pretrained_settings": pretrained_settings["se_resnext50_32x4d"],
    #     "params": {
    #         "out_channels": (3, 64, 256, 512, 1024, 2048),
    #         "block": SEResNeXtBottleneck,
    #         "layers": [3, 4, 6, 3],
    #         "downsample_kernel_size": 1,
    #         "downsample_padding": 0,
    #         "dropout_p": None,
    #         "groups": 32,
    #         "inplanes": 64,
    #         "input_3x3": False,
    #         "num_classes": 1000,
    #         "reduction": 16,
    #     },
    # },
    "se_resnext101_32x4d": {
        "encoder": SENetEncoder,
        "pretrained_settings": pretrained_settings["se_resnext101_32x4d"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": QSEResNeXtBottleneck,
            "layers": [3, 4, 23, 3],
            "downsample_kernel_size": 1,
            "downsample_padding": 0,
            "dropout_p": None,
            "groups": 32,
            "inplanes": 64,
            "input_3x3": False,
            "num_classes": 1000,
            "reduction": 16,
        },
    },
}

def get_quantized_encoder(name, in_channels=3, depth=5, weights=None, output_stride=32, **kwargs):
    """currently only serve senet

    Args:
        name (_type_): _description_
        in_channels (int, optional): _description_. Defaults to 3.
        depth (int, optional): _description_. Defaults to 5.
        weights (_type_, optional): _description_. Defaults to None.
        output_stride (int, optional): _description_. Defaults to 32.

    Raises:
        KeyError: _description_
        KeyError: _description_

    Returns:
        _type_: _description_
    """
    encoders = {}
    encoders.update(senet_encoders)
    try:
        Encoder = encoders[name]["encoder"]
    except KeyError:
        raise KeyError("Wrong encoder name `{}`, supported encoders: {}".format(name, list(encoders.keys())))

    params = encoders[name]["params"]
    params.update(depth=depth)
    encoder = Encoder(**params)

    if weights is not None:
        try:
            settings = encoders[name]["pretrained_settings"][weights]
        except KeyError:
            raise KeyError(
                "Wrong pretrained weights `{}` for encoder `{}`. Available options are: {}".format(
                    weights,
                    name,
                    list(encoders[name]["pretrained_settings"].keys()),
                )
            )
        encoder.load_state_dict(model_zoo.load_url(settings["url"]))

    encoder.set_in_channels(in_channels, pretrained=weights is not None)
    if output_stride != 32:
        encoder.make_dilated(output_stride)

    return encoder