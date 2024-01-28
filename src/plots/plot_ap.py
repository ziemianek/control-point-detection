import matplotlib.pyplot as plt
import json
from src.metrics.metrics import calculate_average_precision, precision, recall
from src.config import METRICS_FILE_PATH

metrics_fig = "metrics_plot.png"

# Load JSON content from a file
with open(METRICS_FILE_PATH, 'r') as file:
    metrics_data = json.load(file)

metrics = []

for iou, metrics_dict in metrics_data:
    iou = float(iou)
    metrics.append(
        {
            'iou': iou,
            'tp': metrics_dict['tp'],
            'fp': metrics_dict['fp'],
            'fn': metrics_dict['fn'],
            'precision': round(precision(metrics_dict['tp'], metrics_dict['fp']), 4),
            'recall': round(recall(metrics_dict['tp'], metrics_dict['fn']), 4),
        }
    )

for entry in metrics:
    print(entry)

precision_values = [entry['precision'] for entry in metrics]
recall_values = sorted([entry['recall'] for entry in metrics])

plt.figure(figsize=(14, 8))
plt.plot(recall_values, precision_values, marker='o', linestyle='-', color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid(True)
plt.savefig(metrics_fig)

average_precision = calculate_average_precision(precision_values, recall_values)
print(f"Average Precision (AP): {average_precision}")
