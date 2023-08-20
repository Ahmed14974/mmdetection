import os
from mmdet.apis import init_detector, inference_detector
import mmcv
import pandas as pd
from PIL import Image
import numpy as np
from mmdet.utils import get_device
import torch
import cv2


def get_tp(pred, target):
    '''
    Calculate the true positives from a segmentation
    model's prediction by comparing with the target.
    Parameters
    ----------
    pred: torch tensor (1, H, W), domain: (-infty:+infty)
        Segmentation model output in logits.
    target: torch tensor (1, H, W), domain: (0,1)
        Segmentation model ground truth.
    Returns
    -------
    iou: float
         true positives count.
    '''
    TP_binary = ((pred == 1).int() + (target == 1).int()) == 2
    TP = float(torch.sum(TP_binary))
    return TP


def get_fp(pred, target):
    '''
    Calculate the false positives from a segmentation
    model's prediction by comparing with the target.
    Parameters
    ----------
    pred: torch tensor (1, H, W), domain: (-infty:+infty)
        Segmentation model output in logits.
    target: torch tensor (1, H, W), domain: (0,1)
        Segmentation model ground truth.
    Returns
    -------
    iou: float
         false positives count.
    '''
    FP_binary = ((pred == 1).int() + (target == 0).int()) == 2
    FP = float(torch.sum(FP_binary))
    return FP


def get_tn(pred, target):
    '''
    Calculate the true negatives from a segmentation
    model's prediction by comparing with the target.
    Parameters
    ----------
    pred: torch tensor (1, H, W), domain: (-infty:+infty)
        Segmentation model output in logits.
    target: torch tensor (1, H, W), domain: (0,1)
        Segmentation model ground truth.
    Returns
    -------
    TN: float
         true negatives count.
    '''
    TN_binary = ((pred == 0).int() + (target == 0).int()) == 2
    TN = float(torch.sum(TN_binary))
    return TN


def get_fn(pred, target):
    '''
    Calculate the false negatives from a segmentation
    model's prediction by comparing with the target.
    Parameters
    ----------
    pred: torch tensor (1, H, W), domain: (-infty:+infty)
        Segmentation model output in logits.
    target: torch tensor (1, H, W), domain: (0,1)
        Segmentation model ground truth.
    Returns
    -------
    FN: float
         false negatives count.
    '''
    FN_binary = ((pred == 0).int() + (target == 1).int()) == 2
    FN = float(torch.sum(FN_binary))
    return FN



VAL_IMGS_PATH = '/deep/group/aicc-bootcamp/cloud-pollution/data/combined_v3_typed_new_composite/COCO_corrected_all_w_null/test/images/'
VAL_CSV_PATH = '/deep/group/aicc-bootcamp/cloud-pollution/data/combined_v3_typed_new_composite/COCO_corrected_all_w_null/test.csv'
# INF_OUT_PATH = '/deep/group/aicc-bootcamp/cloud-pollution/models/sandbox/mahmedc_iseg_cwn_solov2_r101_dcn_fpn_3x_coco/test_inference_images_0.25/'

val_csv = pd.read_csv(VAL_CSV_PATH)
imgs = val_csv.loc[:,'image']
# imgs = sorted(os.listdir(VAL_IMGS_PATH))
masks = val_csv.loc[:,'mask']

# Specify the path to model config and checkpoint file
config_file = '/deep/group/aicc-bootcamp/cloud-pollution/models/sandbox/mahmedc_iseg_cwn_neg_solov2_r101_dcn_fpn_3x_coco_v3/logs/solov2_r101_dcn_fpn_3x_coco.py'
checkpoint_file = '/deep/group/aicc-bootcamp/cloud-pollution/models/sandbox/mahmedc_iseg_cwn_neg_solov2_r101_dcn_fpn_3x_coco_v3/logs/epoch_22.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device=get_device())

print(f"Test set has {len(imgs)} images.")

# thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
thresholds = [0.20]

for threshold in thresholds:
    # print(f"Trying threshold value of: {threshold}")
    pos_iou_list = []
    pos_iou_list_buffed = []
    pos_iou_list_buffed_both = []
    pos_iou_list_int_gt_buffed = []
    pos_iou_optimized_list = []
    TP_list = []
    TN_list = []
    FP_list = []
    FN_list = []
    TP_gt_buffed_list = []
    TN_gt_buffed_list = []
    FP_gt_buffed_list = []
    FN_gt_buffed_list = []
    TP_preds_buffed_list = []
    TN_preds_buffed_list = []
    FP_preds_buffed_list = []
    FN_preds_buffed_list = []

    for i in range(len(imgs)):
        # breakpoint()
        # result = inference_detector(model, VAL_IMGS_PATH + imgs[i])
        result = inference_detector(model, imgs[i])
        # model.show_result(VAL_IMGS_PATH + imgs[i], result, out_file=INF_OUT_PATH + imgs[i], score_thr = 0.25)
        mask = np.load(masks[i])
        mask = mask.astype(np.uint8)
        mask[mask != 0] = 1
        
        tracks = np.zeros(np.asarray(Image.open(imgs[i])).shape[:2])
        if len(result[1][0]) > 0:
            for j in range(len(result[1][0])):
                if result[0][0][j, 4] >= threshold:
                    tracks = np.add(tracks, result[1][0][j])
                # print(j)

        tracks = tracks.astype(np.uint8)
        tracks[tracks != 0] = 1

        kernel_size = 11
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        tracks_buffed = cv2.dilate(tracks, kernel, iterations = 1)
        mask_buffed = cv2.dilate(mask, kernel, iterations=1)

        assert tracks.shape == mask.shape
        assert tracks_buffed.shape == mask_buffed.shape

        tracks_tensor = torch.from_numpy(tracks)
        tracks_tensor_buffed = torch.from_numpy(tracks_buffed)
        mask_tensor = torch.from_numpy(mask)
        mask_tensor_buffed = torch.from_numpy(mask_buffed)

        if np.sum(mask) > 0:
            intersection = tracks & mask
            union = tracks | mask

            pos_iou_list.append(np.sum(intersection) / np.sum(union))

            TP_buffed_both = get_tp(tracks_tensor_buffed, mask_tensor_buffed)
            FP_gt_buffed = get_fp(tracks_tensor, mask_tensor_buffed)
            FN_preds_buffed = get_fn(tracks_tensor_buffed, mask_tensor)
            
            # intersection_buffed = tracks & mask_buffed
            # union_buffed = tracks | mask_buffed
            # intersection_buffed_both = tracks_buffed & mask_buffed
            # union_buffed_both = tracks_buffed | mask_buffed
            intersection_gt_buffed = tracks & mask_buffed
            # pos_iou_list_buffed.append(np.sum(intersection_buffed) / np.sum(union_buffed))
            # pos_iou_list_buffed_both.append(np.sum(intersection_buffed_both) / np.sum(union_buffed_both))
            pos_iou_list_int_gt_buffed.append(np.sum(intersection_gt_buffed) / np.sum(union))
            
            pos_iou_optimized_list.append((TP_buffed_both)/(TP_buffed_both + FN_preds_buffed + FP_gt_buffed))
        

        # TP_list.append(get_tp(tracks_tensor, mask_tensor_buffed))
        # TN_list.append(get_tn(tracks_tensor, mask_tensor_buffed))
        # FP_list.append(get_fp(tracks_tensor, mask_tensor_buffed))
        # FN_list.append(get_fn(tracks_tensor, mask_tensor_buffed))
    
        TP_list.append(get_tp(tracks_tensor, mask_tensor))
        TN_list.append(get_tn(tracks_tensor, mask_tensor))
        FP_list.append(get_fp(tracks_tensor, mask_tensor))
        FN_list.append(get_fn(tracks_tensor, mask_tensor))

        TP_gt_buffed_list.append(get_tp(tracks_tensor, mask_tensor_buffed))
        TN_gt_buffed_list.append(get_tn(tracks_tensor, mask_tensor_buffed))
        FP_gt_buffed_list.append(get_fp(tracks_tensor, mask_tensor_buffed))
        FN_gt_buffed_list.append(get_fn(tracks_tensor, mask_tensor_buffed))

        TP_preds_buffed_list.append(get_tp(tracks_tensor_buffed, mask_tensor))
        TN_preds_buffed_list.append(get_tn(tracks_tensor_buffed, mask_tensor))
        FP_preds_buffed_list.append(get_fp(tracks_tensor_buffed, mask_tensor))
        FN_preds_buffed_list.append(get_fn(tracks_tensor_buffed, mask_tensor))


    assert len(TP_list)  == len(imgs)
    assert len(TN_list)  == len(imgs)
    assert len(FP_list)  == len(imgs)
    assert len(FN_list)  == len(imgs)

    assert len(TP_preds_buffed_list)  == len(imgs)
    assert len(TN_preds_buffed_list)  == len(imgs)
    assert len(FP_preds_buffed_list)  == len(imgs)
    assert len(FN_preds_buffed_list)  == len(imgs)

    assert len(TP_gt_buffed_list)  == len(imgs)
    assert len(TN_gt_buffed_list)  == len(imgs)
    assert len(FP_gt_buffed_list)  == len(imgs)
    assert len(FN_gt_buffed_list)  == len(imgs)
    
    precision = float(sum(TP_list)) / float(sum(TP_list) + sum(FP_list))
    precision_gt_buffed = float(sum(TP_gt_buffed_list)) / float(sum(TP_gt_buffed_list) + sum(FP_gt_buffed_list))
    precision_optimized = float(sum(TP_gt_buffed_list)) / float(sum(TP_gt_buffed_list) + sum(FP_gt_buffed_list))

    sensitivity = float(sum(TP_list)) / float(sum(TP_list) + sum(FN_list))
    sensitivity_gt_buffed = float(sum(TP_gt_buffed_list)) / float(sum(TP_gt_buffed_list) + sum(FN_gt_buffed_list))
    sensitivity_optimized = float(sum(TP_preds_buffed_list)) / float(sum(TP_preds_buffed_list) + sum(FN_preds_buffed_list))

    specificity = float(sum(TN_list)) / float(sum(TN_list) + sum(FP_list))
    specificity_gt_buffed = float(sum(TN_gt_buffed_list)) / float(sum(TN_gt_buffed_list) + sum(FP_gt_buffed_list))
    specificity_optimized = float(sum(TN_gt_buffed_list)) / float(sum(TN_gt_buffed_list) + sum(FP_gt_buffed_list))

    f1_score = float(precision * sensitivity) / float(precision + sensitivity)
    f1_score_gt_buffed = float(precision_gt_buffed * sensitivity_gt_buffed) / float(precision_gt_buffed + sensitivity_gt_buffed)
    f1_score_optimized = float(precision_optimized * sensitivity_optimized) / float(precision_optimized + sensitivity_optimized)

    print(f"The kernel size is {kernel_size}")
    print(f"{checkpoint_file}")
    print(f"With a threshold value of {threshold}, the positive IOU value is {np.mean(pos_iou_list)}")
    # print(f"With a threshold value of {threshold}, the positive IOU buffed value is {np.mean(pos_iou_list_buffed)}")
    # print(f"With a threshold value of {threshold}, the positive IOU buffed (both) value is {np.mean(pos_iou_list_buffed_both)}")
    print(f"With a threshold value of {threshold}, the positive IOU with ground truth buffed in intersection only is {np.mean(pos_iou_list_int_gt_buffed)}")
    print(f"With a threshold value of {threshold}, the positive IOU optimized is {np.mean(pos_iou_optimized_list)}")
    # print(f"With a threshold value of {threshold}, the precision with ground truth buffed is {precision}")
    # print(f"With a threshold value of {threshold}, the sensitivity with ground truth buffed is {sensitivity}")
    # print(f"With a threshold value of {threshold}, the specificity with ground truth buffed is {specificity}")
    # print(f"With a threshold value of {threshold}, the f1-score with ground truth buffed is {f1_score}")
    print(f"With a threshold value of {threshold}, the precision is {precision}")
    print(f"With a threshold value of {threshold}, the precision with ground truth buffed is {precision_gt_buffed}")
    print(f"With a threshold value of {threshold}, the precision optimized is {precision_optimized}")
    print(f"With a threshold value of {threshold}, the sensitivity is {sensitivity}")
    print(f"With a threshold value of {threshold}, the sensitivity with ground truth buffed is {sensitivity_gt_buffed}")
    print(f"With a threshold value of {threshold}, the sensitivity optimized is {sensitivity_optimized}")
    print(f"With a threshold value of {threshold}, the specificity is {specificity}")
    print(f"With a threshold value of {threshold}, the specificity with ground truth buffed is {specificity_gt_buffed}")
    print(f"With a threshold value of {threshold}, the specificity optimized is {specificity_optimized}")
    print(f"With a threshold value of {threshold}, the f1-score is {f1_score}")
    print(f"With a threshold value of {threshold}, the f1-score with ground truth buffed is {f1_score_gt_buffed}")
    print(f"With a threshold value of {threshold}, the f1-score optimized is {f1_score_optimized}")
    print("Done")