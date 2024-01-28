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


def precision(tp: int, fp: int) -> float:
    """
    Calculates precision given the number of true positives (TP) and false positives (FP).

    Parameters:
        tp (int): Number of true positives.
        fp (int): Number of false positives.

    Returns:
        float: Precision value.
    """
    p = 0.0
    try:
        p = min(tp / (tp + fp), 1.0)
    except ZeroDivisionError:
        pass
    return p


def recall(tp: int, fn: int) -> float:
    """
    Calculates recall given the number of true positives (TP) and false negatives (FN).

    Parameters:
        tp (int): Number of true positives.
        fn (int): Number of false negatives.

    Returns:
        float: Recall value.
    """
    r = 0.0
    try:
        r = min(tp / (tp + fn), 1.0)
    except ZeroDivisionError:
        pass
    return r


def calculate_average_precision(precision: List[float], recall: List[float]) -> float:
    """
    Calculates the average precision given precision and recall lists.

    Parameters:
        precision (List[float]): List of precision values.
        recall (List[float]): List of recall values.

    Returns:
        float: Average precision value.
    """
    # Ensure precision and recall lists have the same length
    assert len(precision) == len(recall), "Precision and recall lists must have the same length"

    # Sort precision and recall in decreasing order of recall
    sorted_indices = sorted(range(len(recall)), key=lambda i: recall[i], reverse=True)
    precision = [precision[i] for i in sorted_indices]
    recall = [recall[i] for i in sorted_indices]

    # Initialize variables
    area_under_curve = 0.0
    prev_recall = 0.0

    # Calculate area under the precision-recall curve using the trapezoidal rule
    for i in range(len(precision)):
        area_under_curve += (recall[i] - prev_recall) * precision[i]
        prev_recall = recall[i]

    return abs(area_under_curve)  # Ensure the result is non-negative
