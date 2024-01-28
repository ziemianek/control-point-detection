from typing import List, Union
from torchvision import ops
import torch

def intersection_over_union(
        ground_truth: List[Union[float, int]],
        pred: List[Union[float, int]],
) -> float:
    """
    Calculates the Intersection over Union (IoU) between two bounding boxes.

    Parameters:
        ground_truth (List[Union[float, int]]): List representing the ground truth bounding box,
            containing [x_min, y_min, x_max, y_max].
        pred (List[Union[float, int]]): List representing the predicted bounding box,
            containing [x_min, y_min, x_max, y_max].

    Returns:
        float: Intersection over Union (IoU) score between the two bounding boxes.
    """
    ground_truth = torch.tensor([ground_truth], dtype=torch.float)
    pred = torch.tensor([pred], dtype=torch.float)
    return ops.box_iou(ground_truth, pred).numpy()[0][0]
