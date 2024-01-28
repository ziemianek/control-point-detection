import cv2
import glob
import numpy as np
import torch

from src.nnet.net import Net
from src.metrics.iou import intersection_over_union
from src.common.utils import format_bboxes
from src.metrics.metrics import calculate_metrics
from src.metrics.nms import nms
from src.config import (
    NUM_CLASSES,
    OUTPUTS_DIR,
    TEST_DIR,
    ANNOTATIONS_DIR_PATH,
    MODEL,
    IOU_THRESHOLD
)


def visualize_predictions(image, predictions, targets):
    if len(predictions) == 0:
      return

    # draw the bounding boxes and write the class name on top of it
    for b in nms(predictions):
        best_iou = 0.0
        id, label, score, *box = b
        for gt in targets: 
           iou = intersection_over_union(gt[2:], box)
           if iou > best_iou:
              best_iou = iou
        cv2.rectangle(image,
                    (int(box[0]), int(box[1])),
                    (int(box[2]), int(box[3])),
                    (0, 255, 0), 1)
        cv2.putText(image, f'Prediction (confidence: {(score*100):.2f}%, iou: {best_iou:.2f})',
                    (int(box[0]), int(box[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),
                    2, lineType=cv2.LINE_AA)

    for i, b in enumerate(targets):
      id, label, *box = b
      cv2.rectangle(image,
                  (int(box[0]), int(box[1])),
                  (int(box[2]), int(box[3])),
                  (0, 0, 255), 1)
      cv2.putText(image, f'Target {i+1}/{len(targets)}',
                  (int(box[2]), int(box[3]+5)),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                  2, lineType=cv2.LINE_AA)
    cv2.imwrite(f"{OUTPUTS_DIR}/{image_name}_pred-{len(predictions)}_gts-{len(targets)}.png", image)

if __name__ == "__main__":
  # directory where all the images are present
  test_images = glob.glob(f"{TEST_DIR}/*")

  import xml.etree.ElementTree as ET
  import os
  for root, dirs, files in os.walk(ANNOTATIONS_DIR_PATH):
      image_annotations = [f.split('.')[0] for f in files]

  print(f"Test instances: {len(test_images)}")

  net = Net()
  net.create_model(NUM_CLASSES)
  net.load_model(f"{OUTPUTS_DIR}/{MODEL}")

  net.model.eval()


  for i in range(len(test_images)):

    # get the image file name for saving output later on
    image_name = test_images[i].split('/')[-1].split('.')[0]


    for j in image_annotations:
      if j == image_name:
        annotated_boxes = []
        tree = ET.parse(f'{ANNOTATIONS_DIR_PATH}/{j}.xml')
        root = tree.getroot()

        # box coordinates for xml files are extracted and corrected for image size given
        for member in root.findall('object'):
          # xmin = left corner x-coordinates
          xmin = int(member.find('bndbox').find('xmin').text)
          # xmax = right corner x-coordinates
          xmax = int(member.find('bndbox').find('xmax').text)
          # ymin = left corner y-coordinates
          ymin = int(member.find('bndbox').find('ymin').text)
          # ymax = right corner y-coordinates
          ymax = int(member.find('bndbox').find('ymax').text)

          annotated_boxes.append([xmin, ymin, xmax, ymax])

    image = cv2.imread(test_images[i])
    orig_image = image.copy()

    # BGR to RGB
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)

    # make the pixel range between 0 and 1
    image /= 255.0

    # bring color channels to front
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)

    # convert to tensor
    image = torch.tensor(image, dtype=torch.float).cpu()

    # add batch dimension
    image = torch.unsqueeze(image, 0)

    with torch.no_grad():
        predictions = net.model(image)

    print('------------')
    predictions = format_bboxes(predictions)
    print(f'number of predicted bboxes before nms, image name {image_name}, predictions: {len(predictions)}')
    if len(predictions) > 0:
      predictions = nms(predictions)  # non-maximum suppression
      print(f'number of bboxes after nms, image name {image_name}, predictions: {len(predictions)}')

    for j in range(len(annotated_boxes)):
      annotated_boxes[j].insert(0, 0)
      annotated_boxes[j].insert(1, 1)

    print('number of ground truth boxes: ', len(annotated_boxes))
    metrics = calculate_metrics(predictions, annotated_boxes, IOU_THRESHOLD)
    print("stats per photo: ", metrics)

    visualize_predictions(orig_image, predictions, annotated_boxes)

    print(f"Image {i+1} ({image_name}) done... predicted boxes: {len(predictions)}, real boxes: {len(annotated_boxes)}")
