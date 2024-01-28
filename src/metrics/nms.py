from typing import Union, List
from torchvision import ops
import torch

def nms(
    preds: List[Union[int, float]],
    iou_threshold: float = 0.1,
) -> List[Union[int, float]]:
    """
    Applies Non-Maximum Suppression (NMS) to filter bounding boxes based on their scores.

    Parameters:
        preds (List[Union[int, float]]): List of predictions, where each prediction is represented
            as [class_label, score, x_min, y_min, x_max, y_max].
        iou_threshold (float, optional): IoU threshold for NMS. Defaults to 0.1.

    Returns:
        List[Union[int, float]]: List of filtered predictions after applying NMS.
    """
    boxes = []
    scores = []

    # Extracting boxes and scores from the input predictions
    for p in preds:
        print(p)
        boxes.append(p[3:])
        scores.append(p[1])
    boxes = torch.tensor(boxes, dtype=torch.float)
    scores = torch.tensor(scores, dtype=torch.float)

    # Applying NMS to get indices of selected bounding boxes
    indices = ops.nms(boxes, scores, iou_threshold)

    # Filtering predictions based on NMS indices
    new_boxes = [preds[i] for i in indices]
    return new_boxes
