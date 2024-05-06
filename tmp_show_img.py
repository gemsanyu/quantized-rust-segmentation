import json
import os
import pathlib
from functools import partial

import distinctipy
import pandas as pd
import torch
import torchvision.transforms.v2 as transforms
from PIL import Image
from torch.utils.data import Dataset
from torchvision.ops import box_convert
from torchvision.tv_tensors import BoundingBoxes, Mask
from torchvision.utils import draw_segmentation_masks
from utils import create_polygon_mask, draw_bboxes


class NeaDataset(Dataset):
    def __init__(self, mode="train") -> None:
        super().__init__()
        data_dir = pathlib.Path()/"NEA-Dataset-coco"/mode
        file_names = os.listdir(data_dir.absolute())
        img_names = [file_name for file_name in file_names if ".jpg" in file_name]
        img_paths = [data_dir/img_name for img_name in img_names]
        self.img_list = [Image.open(img_path.absolute()) for img_path in img_paths]
        self.img_dict = {}
        for img_name, img in zip(img_names, self.img_list):
            self.img_dict[img_name] = img
        annotation_file_name = "_annotations.coco.json"
        annotation_file_path = data_dir/annotation_file_name
        with open(annotation_file_path.absolute(), "r") as annotation_file:
            annotation_dict = json.load(annotation_file)
        self.images_df = pd.DataFrame.from_dict(annotation_dict["images"])
        self.annotations_df = pd.DataFrame.from_dict(annotation_dict["annotations"])
        self.categories_df = pd.DataFrame.from_dict(annotation_dict["categories"])
        self.annotations_df['label'] = self.annotations_df['category_id'].apply(lambda x: self.categories_df.loc[x]['name'])
        self.annotations_df = pd.merge(self.annotations_df, self.images_df, left_on="image_id", right_on="id")
        self.annotations_df["image_id"] = self.annotations_df["file_name"]
        self.annotations_df.drop('id_x', axis=1, inplace=True)
        self.annotations_df.drop('id_y', axis=1, inplace=True)
        self.annotations_df.set_index('image_id', inplace=True)
        self.categories = self.annotations_df["label"].unique().tolist()
        self.annotations_df = self.annotations_df.groupby("image_id").agg({
            "bbox":list,
            "area":list,
            "segmentation":list,
            "iscrowd":list,
            "label":list,
            "category_id":list,
            "license":"first",
            "file_name":"first",
            "height":"first",
            "width":"first",
            "date_captured":"first",
        })
        self.annotations_df.rename(columns={
            "bbox":"bboxes",
            "label":"labels",
            "segmentation":"segmentations",
            "area":"areas"
        }, inplace=True)
        colors = distinctipy.get_colors(len(self.categories))
        colors = [tuple(int(c*255) for c in color) for color in colors]
        
        
        sample_img_name = img_names[0]
        sample_img = self.img_dict[sample_img_name]
        polygon_points = self.annotations_df.loc[sample_img_name]["segmentations"]
        mask_imgs = [create_polygon_mask(sample_img.size, polygon[0]) for polygon in polygon_points]
        pil_to_tensor_transform = transforms.PILToTensor()
        masks = torch.concat([Mask(pil_to_tensor_transform(mask_img), dtype=bool) for mask_img in mask_imgs])
        labels = self.annotations_df.loc[sample_img_name]['labels']
        bboxes = self.annotations_df.loc[sample_img_name]['bboxes']
        annotated_tensor = draw_segmentation_masks(
            image=pil_to_tensor_transform(sample_img),
            masks=masks,
            alpha=0.3,
            colors=[colors[i] for i in [self.categories.index(label) for label in labels]]
        )
        annotated_tensor = draw_bboxes(
            image=annotated_tensor,
            boxes=box_convert(torch.tensor(bboxes), "xywh", "xyxy"),
            labels=labels,
            colors=[colors[i] for i in [self.categories.index(label) for label in labels]]
        )
        img = transforms.ToPILImage()(annotated_tensor)
        img.show()