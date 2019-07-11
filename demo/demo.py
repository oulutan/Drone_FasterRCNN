from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
import cv2

config_file = "e2e_faster_rcnn_X_101_32x8d_FPN_1x_custom.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
#cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
)
# load image and then run prediction
image = cv2.imread('test_img.jpg')
predictions = coco_demo.run_on_opencv_image(image)
cv2.imshow('Predictions', predictions)
cv2.waitKey(0)
