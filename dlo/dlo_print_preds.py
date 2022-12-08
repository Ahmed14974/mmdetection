import mmcv
from mmcv.runner import load_checkpoint

from mmdet.apis import inference_detector, show_result_pyplot
from mmdet.models import build_detector

# Choose to use a config and initialize the detector
config = '/deep/u/yuzu/aicc-win21-cloud-features/mmdetection/configs/solov2/solov2_x101_dcn_fpn_3x_coco.py'
# Setup a checkpoint file to load
# checkpoint = "/deep/group/aicc-bootcamp/cloud-pollution/models/sandbox/mahmedc_iseg_solov2_x101_dcn_fpn_3x_coco/checkpoints/solov2_x101_dcn_fpn_3x_coco_20220513_214337-aef41095.pth"
checkpoint = "/deep/group/aicc-bootcamp/cloud-pollution/models/sandbox/mahmedc_iseg_solov2_x101_dcn_fpn_3x_coco/logs/epoch_17.pth"
# Set the device to be used for evaluation
device='cuda:0'

# Load the config
config = mmcv.Config.fromfile(config)
# Set pretrained to be None since we do not need pretrained model here
config.model.pretrained = None

# Initialize the detector
model = build_detector(config.model)

# Load checkpoint
checkpoint = load_checkpoint(model, checkpoint, map_location=device)

# Set the classes of models for inference
model.CLASSES = checkpoint['meta']['CLASSES']

# We need to set the model's cfg for inference
model.cfg = config

# Convert the model to GPU
model.to(device)
# Convert the model into evaluation mode
model.eval()

img = mmcv.imread('/deep/group/aicc-bootcamp/cloud-pollution/data/'\
                        'combined_v3_typed_new_composite/COCO_format_cropped/'\
                        'train/images/mod2002121.1920D_crop_0.png')
result = inference_detector(model, img)
show_result_pyplot(model, img, result, score_thr=0.1, out_file='yuzu_test_solov2_result.jpg')
model.show_result(img, result, out_file='yuzu_test_solov2_result_boxes.jpg')