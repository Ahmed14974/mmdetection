import mmcv
from mmdet.apis import init_detector, inference_detector
import pandas as pd
from PIL import Image
import numpy as np
from mmdet.utils import get_device

VAL_IMGS_PATH = '/deep/group/aicc-bootcamp/cloud-pollution/data/combined_v3_typed_new_composite/COCO_format_cropped/val/images'
VAL_CSV_PATH = '/deep/group/aicc-bootcamp/cloud-pollution/data/combined_v3_typed_new_composite/COCO_format_cropped/val.csv'
DLO_IMG_DIR = 'dlo/dlo_test_imgs/dlo_from_preds_imgs'

val_csv = pd.read_csv(VAL_CSV_PATH)
imgs = val_csv.loc[:, 'image']
masks = val_csv.loc[:, 'mask']

config_file = '/deep/group/aicc-bootcamp/cloud-pollution/models/sandbox/mahmedc_iseg_cropped_solov2_light_r18_fpn_3x_coco/logs/solov2_light_r18_fpn_3x_coco.py'
checkpoint_file = '/deep/group/aicc-bootcamp/cloud-pollution/models/sandbox/mahmedc_iseg_cropped_solov2_light_r18_fpn_3x_coco/logs/latest.pth'

model  = init_detector(config_file, checkpoint_file, device=get_device())
print(f"Val set has {len(imgs)} images")

# thresholds = [0.05 * x for x in range(1, 7)]
thresholds = [0.01 * x for x in range(5, 15)]
pos_ious = []
counter = 0
for threshold in thresholds:
    print(f"trying threshold value {threshold}")
    pos_iou_list = []
    for i in range(len(imgs)):
        result = inference_detector(model, imgs[i])
        mask = np.load(masks[i])
        mask = mask.astype(np.uint8)
        mask[mask != 0] = 1

        if np.sum(mask) > 0:
            tracks = np.zeros(np.asarray(Image.open(imgs[i])).shape[:2])
            if len(result[1][0]) > 0:
                for j in range(len(result[1][0])):
                    if result[0][0][j,4] >= threshold:
                        tracks = np.add(tracks, result[1][0][j])
            tracks = tracks.astype(np.uint8)
            tracks[tracks != 0] = 1
            assert tracks.shape == mask.shape
            
            # if i > 25 and counter < 20 and np.sum(tracks) > 0:
            # if i == 52:
            #     Image.fromarray(tracks * 255).save(f"{DLO_IMG_DIR}/{i}_pred.png")
            #     Image.fromarray(mask * 255).save(f"{DLO_IMG_DIR}/{i}_truth.png")
            #     Image.open(imgs[i]).save(f"{DLO_IMG_DIR}/{i}_image_orig.png")
            #     np.savetxt(f"{DLO_IMG_DIR}/{i}_pred.csv", tracks * 255, fmt="%u", delimiter=",")
            #     np.savetxt(f"{DLO_IMG_DIR}/{i}_truth.csv", mask * 255, fmt="%u", delimiter=",")
            #     counter += 1
            #     break

            intersection = tracks & mask
            union = tracks | mask
            pos_iou_list.append(np.sum(intersection) / np.sum(union))
    pos_ious.append(np.mean(pos_iou_list))

print(f"pos_ious for threshold {thresholds} is {pos_ious}")    
# print(f"With threshold {threshold}, pos iou is {np.mean(pos_iou_list)}")



