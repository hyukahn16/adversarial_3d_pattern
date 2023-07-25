import utils
from utils import plot_boxes
from arch.yolov3_models import YOLOv3Darknet

import os
from PIL import Image
import numpy as np
from torchvision import transforms
from torchvision.ops import box_iou
from tqdm import tqdm

# 1. GET AND SAVE LABELS FROM YOLO
device = "cuda:0"
model = YOLOv3Darknet().eval().to(device)
model.load_darknet_weights('arch/weights/yolov3.weights')

conf_thresh = 0.5 # FIXME
architecture = "yolov3"

dir = "./test_images" # where the testing images are
if not os.path.exists(dir):
  print("Created ./test_images directory")
  os.makedirs(dir)
labels_dir = os.path.join(dir, "yolo-labels") # where the testing images labels are
if not os.path.exists(labels_dir):
   os.makedirs(labels_dir)

# Get yolo predictions
print("Starting yolov3 predictions")
img_files = os.listdir(dir)
for _, img_f in tqdm(enumerate(img_files), total=len(img_files)):
  img_path = os.path.join(dir, img_f)

  # Check if image file or not
  if not os.path.isfile(img_path):
    continue
    
  # Load image file as tensor
  test_img = Image.open(img_path).convert('RGB')
  convert_tensor = transforms.ToTensor()
  test_img = convert_tensor(test_img).to(device)
  test_img = test_img[None, :] # Turn dim to batch dim

  # Get the labels for the testing images
  output = model(test_img)
  
  person_label = 0
  output = utils.get_region_boxes_general(output, model, conf_thresh=conf_thresh, name=architecture)
  for i, boxes in enumerate(output):
    # Create and open label file
    label_file_name = img_f.rsplit(".", 1)[0] + ".txt"
    label_file_dir = os.path.join(labels_dir, label_file_name)
    label_f = open(label_file_dir, 'w') # Create and open file

    # Skip this image if no boxes were predicted
    if len(boxes) == 0:
        label_f.close()
        continue
    assert boxes.shape[1] == 7
  
    # Get bounding box information
    test_nms_thresh = 1.0 # FIXME:??? 1.0???
    boxes = utils.nms(boxes, nms_thresh=test_nms_thresh)
    x_centers = boxes[..., 0]
    y_centers = boxes[..., 1]
    widths = boxes[..., 2]
    heights = boxes[..., 3]
    det_confs = boxes[..., 4]
    labels = boxes[..., 6]

    # Save box info to label file
    for i in range(len(boxes)):
      if labels[i] == person_label:
        label_f.write(f'{labels[i]} {x_centers[i]} {y_centers[i]} {widths[i]} {heights[i]} {det_confs[i]}\n') 
    label_f.close()

###############################################################################
###############################################################################
###############################################################################
# 2. PLOT LABELS
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
        if not boxes:
           continue
        boxes = np.array([b.strip('\n').split() for b in boxes], dtype=float)
        boxes = [[b[1], b[2], b[3], b[4], b[5], b[0]] for b in boxes]
        plot_boxes(img, boxes, savename=os.path.join(plot_dir, img_name))
