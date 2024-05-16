import random
import ssl

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.ao.quantization
from arguments import prepare_args
from segmentation_dataset import (SegmentationDataset, get_preprocessing,
                                  get_validation_augmentation)
from segmentation_models_pytorch.utils.metrics import (Accuracy, Fscore, IoU,
                                                       Precision, Recall)
from segmentation_models_pytorch.base.model import SegmentationModel
from segmentation_models_pytorch.utils import losses
from segmentation_models_pytorch.utils.train import ValidEpoch
from torch.utils.data import DataLoader, Dataset

from setup import setup

def quantize_model(model:SegmentationModel, backend = "fbgemm"):
    model.qconfig = torch.ao.quantization.get_default_qconfig(backend)
    model_q = torch.quantization.prepare(model, inplace=False)
    model_q = torch.ao.quantization.convert(model_q, inplace=False)
    return model_q


def get_test_dataset(args)->Dataset:
    preprocessing_fn = smp.encoders.get_preprocessing_fn(args.encoder, args.encoder_pretrained_source)
    preprocessing = get_preprocessing(preprocessing_fn)
    validation_augmentation = get_validation_augmentation()
    test_dataset = SegmentationDataset(name=args.dataset, mode="test", augmentation=validation_augmentation, preprocessing=preprocessing)
    return test_dataset
    
def test(args):
    """main function to run the training, calling setup and preparing the train/validation epoch

    Args:
        args (_type_): _description_
    """
    model, _, _, _, _ = setup(args, load_best=True, is_quantized=True)
    model_q = quantize_model(model)    
    test_dataset = get_test_dataset(args)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)
    loss = losses.JaccardLoss()
    metrics = [IoU(), Accuracy(), Precision(), Recall(), Fscore()]
    tester = ValidEpoch(model_q, loss, metrics, device=args.device, verbose=True)
    test_log = tester.run(test_dataloader)
    print(test_log)

if __name__ == "__main__":
    ssl._create_default_https_context = ssl._create_unverified_context
    args = prepare_args()
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    test(args)