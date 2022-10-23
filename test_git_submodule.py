import torch, torchvision
import mmdet
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
import mmcv
import os.path as osp
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.apis import inference_detector, show_result_pyplot, set_random_seed
from mmdet.models import build_detector
from mmdet.datasets import build_dataset
from mmdet.apis import train_detector
from mmdet.utils import get_device


train_annotation = mmcv.load('/deep/group/aicc-bootcamp/cloud-pollution/data/'\
                        'combined_v3_typed_new_composite/COCO_format_cropped/'\
                        'train/train_annotation_coco.json')
print(type(train_annotation))
cfg = Config.fromfile('./configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py')
cfg.device = get_device()

# Modify dataset type and path
cfg.dataset_type = 'COCODataset'

cfg.data.test.ann_file = '/deep/group/aicc-bootcamp/cloud-pollution/data/'\
                        'combined_v3_typed_new_composite/COCO_format_cropped/'\
                        'test/test_annotation_coco.json'
cfg.data.test.img_prefix = '/deep/group/aicc-bootcamp/cloud-pollution/data/'\
                        'combined_v3_typed_new_composite/COCO_format_cropped/'\
                        'test/images/'
cfg.data.test.classes = ('shiptrack',) #???

cfg.data.train.ann_file = '/deep/group/aicc-bootcamp/cloud-pollution/data/'\
                        'combined_v3_typed_new_composite/COCO_format_cropped/'\
                        'train/train_annotation_coco.json'
cfg.data.train.img_prefix = '/deep/group/aicc-bootcamp/cloud-pollution/data/'\
                        'combined_v3_typed_new_composite/COCO_format_cropped/'\
                        'train/images/'
cfg.data.train.classes = ('shiptrack',)


cfg.data.val.ann_file = '/deep/group/aicc-bootcamp/cloud-pollution/data/'\
                        'combined_v3_typed_new_composite/COCO_format_cropped/'\
                        'val/val_annotation_coco.json'
cfg.data.val.img_prefix = '/deep/group/aicc-bootcamp/cloud-pollution/data/'\
                        'combined_v3_typed_new_composite/COCO_format_cropped/'\
                        'val/images/'
cfg.data.val.classes = ('shiptrack',)

# modify num classes of the model in box head and mask head
cfg.model.roi_head.bbox_head.num_classes = 1
cfg.model.roi_head.mask_head.num_classes = 1

# We can still the pre-trained Mask RCNN model to obtain a higher performance
cfg.load_from = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'

# Set up working dir to save files and logs.
cfg.work_dir = './test_maskrcnn'

# The original learning rate (LR) is set for 8-GPU training.
# We divide it by 8 since we only use one GPU.
cfg.optimizer.lr = 0.02 / 8
cfg.lr_config.warmup = None
cfg.log_config.interval = 10

# We can set the evaluation interval to reduce the evaluation times
cfg.evaluation.interval = 12
# We can set the checkpoint saving interval to reduce the storage cost
cfg.checkpoint_config.interval = 12

# Set seed thus the results are more reproducible
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)

# We can also use tensorboard to log the training process
cfg.log_config.hooks = [
    dict(type='TextLoggerHook'),
    dict(type='TensorboardLoggerHook')]

# We can initialize the logger for training and have a look
# at the final config used for training
print(f'Config:\n{cfg.pretty_text}')

# Build dataset
datasets = [build_dataset(cfg.data.train)]

# Build the detector
model = build_detector(cfg.model)

# Add an attribute for visualization convenience
model.CLASSES = datasets[0].CLASSES

# Create work_dir
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
train_detector(model, datasets, cfg, distributed=False, validate=True)

img = mmcv.imread('/deep/group/aicc-bootcamp/cloud-pollution/data/'\
                        'combined_v3_typed_new_composite/COCO_format_cropped/'\
                        'train/images/mod2002121.1920D_crop_0.png')

model.cfg = cfg
result = inference_detector(model, img)
show_result_pyplot(model, img, result, out='test_maskrcnn_result.jpg')