import random
import ssl

import numpy as np
import segmentation_models_pytorch as smp
import torch
from arguments import prepare_args
from segmentation_dataset import (SegmentationDataset, get_preprocessing,
                                  get_validation_augmentation)
from segmentation_models_pytorch.utils.metrics import (Accuracy, Fscore, IoU,
                                                       Precision, Recall)
from segmentation_models_pytorch.utils import losses
from segmentation_models_pytorch.utils.train import ValidEpoch
from torch.utils.data import DataLoader, Dataset

from setup import setup


def get_test_dataset(args)->Dataset:
    preprocessing_fn = smp.encoders.get_preprocessing_fn(args.encoder,args.encoder_pretrained_source)
    preprocessing = get_preprocessing(preprocessing_fn)
    validation_augmentation = get_validation_augmentation()
    test_dataset = SegmentationDataset(name=args.dataset, mode="test", augmentation=validation_augmentation, preprocessing=preprocessing)
    return test_dataset
    
def test(args):
    """main function to run the training, calling setup and preparing the train/validation epoch

    Args:
        args (_type_): _description_
    """
    model, optimizer, tb_writer, _, _ = setup(args, load_best=True)
    test_dataset = get_test_dataset(args)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)
    loss = losses.JaccardLoss()
    metrics = [IoU(), Accuracy(), Precision(), Recall(), Fscore()]
    tester = ValidEpoch(model, loss, metrics, device=args.device, verbose=True)
    test_log = tester.run()
    print(test_log)

if __name__ == "__main__":
    ssl._create_default_https_context = ssl._create_unverified_context
    args = prepare_args()
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    test(args)