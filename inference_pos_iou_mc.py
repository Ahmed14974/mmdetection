import os
from mmdet.apis import init_detector, inference_detector
import mmcv
import pandas as pd
from PIL import Image
import numpy as np
from mmdet.utils import get_device

VAL_IMGS_PATH = '/deep/group/aicc-bootcamp/cloud-pollution/data/combined_v3_typed_new_composite/COCO_corrected_all_w_null/val/images/'
VAL_CSV_PATH = '/deep/group/aicc-bootcamp/cloud-pollution/data/combined_v3_typed_new_composite/COCO_corrected_all_w_null/val.csv'

val_csv = pd.read_csv(VAL_CSV_PATH)
imgs = val_csv.loc[:,'image']
masks = val_csv.loc[:,'mask']

# Specify the path to model config and checkpoint file
config_file = '/deep/group/aicc-bootcamp/cloud-pollution/models/sandbox/yuzu_iseg_cwn_solov2_r50_fpn_3x_coco_filter_0.07/configs/solov2_r50_fpn_3x_coco.py'
checkpoint_file = '/deep/group/aicc-bootcamp/cloud-pollution/models/sandbox/yuzu_iseg_cwn_solov2_r50_fpn_3x_coco/logs/epoch_22.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device=get_device())

print(f"Validation set has {len(imgs)} images.")

thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

for threshold in thresholds:
    print(f"Trying threshold value of: {threshold}")
    pos_iou_list = []

    for i in range(len(imgs)):
        result = inference_detector(model, imgs[i])
        model.show_result(imgs[i], result, out_file=f"/deep/group/aicc-bootcamp/cloud-pollution/models/sandbox/yuzu_iseg_cwn_solov2_r50_fpn_3x_coco_filter_0.07/logs/images/{i}.png", score_thr=0.25)
        mask = np.load(masks[i])
        mask = mask.astype(np.uint8)
        mask[mask != 0] = 1
        
        if np.sum(mask) > 0:
            tracks = np.zeros(np.asarray(Image.open(imgs[i])).shape[:2])
            if len(result[1][0]) > 0:
                for j in range(len(result[1][0])):
                    if result[0][0][j, 4] >= threshold:
                        tracks = np.add(tracks, result[1][0][j])
                    # print(j)

            tracks = tracks.astype(np.uint8)
            tracks[tracks != 0] = 1

            assert tracks.shape == mask.shape

            intersection = tracks & mask
            union = tracks | mask

            pos_iou_list.append(np.sum(intersection) / np.sum(union))

    print(f"With a threshold value of {threshold}, the positive IOU value is {np.mean(pos_iou_list)}")
