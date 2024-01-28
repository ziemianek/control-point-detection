import json
import matplotlib.pyplot as plt
import numpy as np


def precision(tp, fp):
    p = 0.0
    try:
        p = min(tp / (tp + fp), 1.0)
    except ZeroDivisionError:
        pass
    return p


def recall(tp, fn):
    r = 0.0
    try:
        r = min(tp / (tp + fn), 1.0)
    except ZeroDivisionError:
        pass
    return r


def calculate_average_precision(precision, recall):
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



dir = "/Users/ziemian/Code/bt/paper"
metrics_file = f"{dir}/values.txt"
metrics_fig = "metrics_plot.png"

# Load JSON content from a file
with open(metrics_file, 'r') as file:
    content = file.readlines()

ious = [round(float(i), 1) for i in np.arange(0.0, 1.1, 0.1)]
# ious = [0.25, 0.5, 0.75]
metrics = []

for iou in ious:
    values = []
    for line in content:
        if line.startswith(f'{iou}'):
            values.append(line.split()[1])
    iou = float(iou)
    values[0] = int(values[0])
    values[1] = int(values[1])
    values[2] = int(values[2])
    metrics.append(
        {
            'iou': iou,
            'tp': values[0],
            'fp': values[1],
            'fn': values[2],
            'precision': round(precision(values[0], values[1]), 4),
            'recall': round(recall(values[0], values[2]), 4),
        }
    )

for iou in ious:
    for entry in metrics:
        print(entry) if entry['iou'] == iou else None

import numpy as np

p = [entry['precision'] for entry in metrics]
r = sorted([entry['recall'] for entry in metrics])

print(p)
print(r)

# # Smooth the Precision-Recall curve
# smooth_p, smooth_r = smooth_precision_recall_curve(p, r)


# Plot the smoothed curve
plt.figure(figsize=(14, 8))
plt.plot(r, p, marker='o', linestyle='-', color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Krzywa Precision-Recall')
plt.grid(True)
# plt.show()
# plt.savefig(f"{dir}/{metrics_fig}")

average_precision = calculate_average_precision(p, r)
print(f"Average Precision (AP): {average_precision}")
