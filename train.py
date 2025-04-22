import os
import yaml
import random
import numpy as np
import pandas as pd
from datetime import datetime
import torch
from ultralytics import settings, YOLO
settings.update({'datasets_dir': './'})

# import ultralytics.data.build as build
# from ultralytics.data.dataset import YOLOConcatDataset

# _original_build = build.build_yolo_dataset

# def _build_double(cfg, img_path, batch, data, mode="train", rect=False, stride=32, multi_modal=False):
#     if mode == "train":
#         ds_orig = _original_build(cfg, img_path, batch, data, mode, rect=rect, stride=stride, multi_modal=multi_modal)
#         ds_aug  = _original_build(cfg, img_path, batch, data, mode, rect=rect, stride=stride, multi_modal=multi_modal)
#         ds_orig.augment = False
#         return YOLOConcatDataset([ds_orig, ds_aug])
    
#     return _original_build(cfg, img_path, batch, data, mode, rect, stride, multi_modal)

# build.build_yolo_dataset = _build_double

def train_model(ex_dict):
    ex_dict['Train Time'] = datetime.now().strftime("%y%m%d_%H%M%S")
    name = f"{ex_dict['Train Time']}_{ex_dict['Model Name']}_{ex_dict['Dataset Name']}_Iter_{ex_dict['Iteration']}"
    task = f"{ex_dict['Experiment Time']}_Train"
    
    ex_dict['Train Results'] = ex_dict['Model'].train(
        model = f"{ex_dict['Model Name']}.yaml",
        name=name,
        data=ex_dict['Data Config'] ,
        epochs=ex_dict['Epochs'],
        imgsz=ex_dict['Image Size'],
        batch=ex_dict['Batch Size'],
        patience=20,
        save=True,
        device=ex_dict['Device'],
        exist_ok=True,
        verbose=False,
        optimizer=ex_dict['Optimizer'],
        lr0=ex_dict['lr0'],  
        lrf=ex_dict.get('lrf', 1),
        weight_decay = ex_dict['Weight Decay'],
        momentum = ex_dict['Momentum'],
        pretrained=False,
        amp=False,
        task=task,
        project=f"{ex_dict['Output Dir']}",

        augment=True,
        mosaic=ex_dict.get('mosaic', 1.0),
        mixup=ex_dict.get('mixup', 0.5),
        hsv_h=ex_dict.get('hsv_h', 0.015),
        hsv_s=ex_dict.get('hsv_s', 1.0),
        hsv_v=ex_dict.get('hsv_v', 0.6),
        degrees=ex_dict.get('degrees', 15),
        translate=ex_dict.get('translate', 0.3),
        scale=ex_dict.get('scale', 0.75),
        shear=ex_dict.get('shear', 5.0),
        fliplr=ex_dict.get('fliplr', 0.5),
        flipud=ex_dict.get('flipud', 0.0)
    )
    pt_path = f"{ex_dict['Output Dir']}/{name}/weights/best.pt"
    ex_dict['PT path'] = pt_path
    ex_dict['Model'].load(pt_path)
    return ex_dict