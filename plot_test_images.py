import utils
# from utils import plot_boxes, plot_boxes_cv2

import os
from PIL import Image
import numpy as np
from torchvision.ops import box_iou
# from avg_precision import get_AP

dir = "./test_images" # where the testing images are
labels_dir = os.path.join(dir, "yolo-labels") # where the testing images labels are

device = "cuda:0"
model = YOLOv3Darknet().eval().to(device)
model.load_darknet_weights('arch/weights/yolov3.weights')

conf_thresh = 0.5 # FIXME
architecture = "yolov3

dir = "./test_images" # where the testing images are
labels_dir = os.path.join(dir, "yolo-labels") # where the testing images labels are
for f in os.listdir(dir):
  if not os.path.isfile(os.path.join(dir, f)):
    continue
    
  # TODO: Load image file as tensor
  test_img = None
  # Get the labels for the testing images
  output = self.model(test_img)
  
  output = utils.get_region_boxes_general(output, model, conf_thresh=conf_thresh, name=architecture)
  for i, boxes in enumerate(output):
      if len(boxes) == 0:
          pos.append((0.0, False))
          continue
      assert boxes.shape[1] == 7
    
      test_nms_thresh = 1.0 # FIXME:??? 1.0???
      boxes = utils.nms(boxes, nms_thresh=test_nms_thresh)
      w1 = boxes[..., 0] - boxes[..., 2] / 2
      h1 = boxes[..., 1] - boxes[..., 3] / 2
      w2 = boxes[..., 0] + boxes[..., 2] / 2
      h2 = boxes[..., 1] + boxes[..., 3] / 2
      # bboxes = torch.stack([w1, h1, w2, h2], dim=-1)
      # bboxes = bboxes.view(-1, 4).detach() * self.img_size
      scores = boxes[..., 4]
      labels = boxes[..., 6]
  
      # FIXME: there can be multiple boxes
      person_label = 0
      for i in range(len(boxes):
        if labels[i] == person_label:
          # Save box info to label file
          label_file_name = f.split(".")[0] + ".txt"
          label_file_dir = os.path.join(labels_dir, label_file_name)
          # if not os.path.exists(label_file_dir):
          #   # Create file
          with open(label_file_dir, 'w') as label_f:
            label_f.write(...) # FIXME

# 2. PLOT LABELS #
plot_dir = os.path.join(dir, "plotted") # where to save plotted images

# Find all patched image names
patched_imgs = [f for f in os.listdir(dir) 
             if os.path.isfile(os.path.join(dir, f))]

# Make sure plot saving folder exists
if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)

# Draw boxes
for img_name in patched_imgs:
    img_dir = os.path.join(dir, img_name)
    label_dir = os.path.join(labels_dir, img_name.split(".")[0] + ".txt")
    img = Image.open(img_dir).convert('RGB')
    with open(label_dir) as label:
        boxes = label.readlines()
        boxes = np.array([b.strip('\n').split() for b in boxes], dtype=float)
        boxes = [[b[0], b[2], b[1], b[4], b[3]] for b in boxes]
        plot_boxes(img, boxes, savename=os.path.join(plot_dir, img_name))
