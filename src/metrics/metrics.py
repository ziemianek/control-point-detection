from typing import List, Dict, Union
from src.metrics.iou import intersection_over_union


def calculate_metrics(
    predicted_boxes: List[Union[int, float]],
    ground_truth_boxes: List[Union[int, float]],
    iou_threshold: float
) -> Dict[str, int]:
    """
    Calculates evaluation metrics for object detection.

    Parameters:
        predicted_boxes (List[Union[int, float]]): List of predicted bounding boxes,
            each represented as [class_label, score, x_min, y_min, x_max, y_max].
        ground_truth_boxes (List[Union[int, float]]): List of ground truth bounding boxes,
            each represented as [class_label, x_min, y_min, x_max, y_max].
        iou_threshold (float): Intersection over Union (IoU) threshold for considering a match.

    Returns:
        Dict[str, int]: Dictionary containing true positives (TP), false positives (FP),
            and false negatives (FN).
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # List to keep track of whether each ground truth box has been matched
    gt_matched = [False] * len(ground_truth_boxes)

    # Iterate over each predicted box
    for pred_box in predicted_boxes:
        pred_matched = False

        # Check for matches with ground truth boxes
        for i, gt_box in enumerate(ground_truth_boxes):
            if gt_matched[i]:
                continue  # Skip if ground truth box already matched

            iou = intersection_over_union(gt_box[2:], pred_box[3:])

            # If IoU is above the threshold, consider it a true positive
            if iou >= iou_threshold:
                true_positives += 1
                pred_matched = True
                gt_matched[i] = True  # Mark ground truth box as matched
                break

        # If no match found, consider it a false positive
        if not pred_matched:
            false_positives += 1

    # Count remaining unmatched ground truth boxes as false negatives
    false_negatives = len(ground_truth_boxes) - sum(gt_matched)

    return {
        'TP': true_positives,
        'FP': false_positives,
        'FN': false_negatives,
    }
