import pathlib
import os
import random

from torchvision import models, datasets

import numpy as np
import torch
import torchvision.transforms.v2 as transforms
from PIL import Image, ImageDraw
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
    
def run():
    dataset_dir = pathlib.Path("")/"NEA-Dataset-coco"/"train"
    test_dataset_dir =pathlib.Path("")/"NEA-Dataset-coco"/"test"
    annotation_file_path = dataset_dir/"_annotations.coco.json"
    test_annotation_file_path = test_dataset_dir/"_annotations.coco.json"
    register_coco_instances("ndea_dataset_train", {}, annotation_file_path.absolute(), dataset_dir.absolute())
    register_coco_instances("ndea_dataset_test", {}, test_annotation_file_path.absolute(), test_dataset_dir.absolute())
    
    ndea_metadata = MetadataCatalog.get("ndea_dataset_train")
    dataset_dicts = DatasetCatalog.get("ndea_dataset_train")
    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("ndea_dataset_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 1500    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    
    # os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()
    trainer.register_hooks()
    
    
    # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
           # set a custom testing threshold
    predictor = DefaultPredictor(cfg)
    
    test_dataset_dicts = DatasetCatalog.get("ndea_dataset_train")

    
    
    for d in random.sample(test_dataset_dicts, 5):
        img = Image.open(d["file_name"])
        img.show()
        img = np.asarray(img)[:,:,::-1]
        # print(img.shape)
        # exit()
        # img.transpose(2,1,0)
        outputs = predictor(img)
        # print(outputs["instances"].to("cpu"))
        # # print(outputs)
        # # exit()
        v = Visualizer(img, metadata=ndea_metadata, scale=0.5)
        # exit()
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        out_img = out.get_image()
        out_img = np.asarray(out_img)
        out_img = out_img[:,:,::-1]
        # # out_img.transpose(2,1,0)
        out_img = Image.fromarray(out_img)
        out_img.show()

if __name__ == "__main__":
    run()