import matplotlib.pyplot as plt
import numpy as np
from src.metrics.metrics import *
from src.config import METRICS_FILE_PATH
dir = "/Users/ziemian/Code/bt/paper"
metrics_file = f"{dir}/values.txt"
metrics_fig = "metrics_plot.png"

# Load JSON content from a file
with open(metrics_file, 'r') as file:
    content = file.readlines()

ious = [round(float(i), 1) for i in np.arange(0.0, 1.1, 0.1)]
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


p = [entry['precision'] for entry in metrics]
r = sorted([entry['recall'] for entry in metrics])


# Plot the curve
plt.figure(figsize=(14, 8))
plt.plot(r, p, marker='o', linestyle='-', color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Krzywa Precision-Recall')
plt.grid(True)
plt.savefig(f"{dir}/{metrics_fig}")

average_precision = calculate_average_precision(p, r)
print(f"Average Precision (AP): {average_precision}")
