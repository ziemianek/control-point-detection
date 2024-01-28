import json
import numpy as np
import torch
import tqdm
from src.config import DEVICE, OUTPUTS_DIR, MODEL
from src.common.utils import format_bboxes
from src.metrics.metrics import calculate_metrics
from src.metrics.nms import nms
from src.nnet.net import Net


def load_model(net: Net, path: str) -> None:
    """
    Loads the model weights from a given file path.

    Parameters:
        net (Net): Neural network model.
        path (str): Path to the model checkpoint file.

    Returns:
        None
    """
    if not torch.cuda.is_available():
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(path)
    net.model.load_state_dict(checkpoint)


def eval(net: Net) -> None:
    """
    Evaluates the neural network on the validation dataset and saves metrics to a JSON file.

    Parameters:
        net (Net): Neural network model.

    Returns:
        None
    """
    load_model(net, f'{OUTPUTS_DIR}/{MODEL}')

    net.model.eval()

    all_metrics = []
    for iou in [round(float(i), 2) for i in np.arange(0.0, 1.1, 0.1)]:
        tp = 0
        fp = 0
        fn = 0
        for images, targets in tqdm.tqdm(net.valid_loader, total=len(net.valid_loader)):
            images = [image.to(DEVICE) for image in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            with torch.no_grad():
                predictions = net.model(images)

            predictions = format_bboxes(predictions)
            targets = format_bboxes(targets)
            
            predictions = nms(predictions, 0.01)  # if one detection collides with another with at least 0.1 IOU, then remove it

            metrics = calculate_metrics(predictions, targets, iou)
            all_metrics.append([iou, metrics])

            tp += metrics['TP']
            fp += metrics['FP']
            fn += metrics['FN']

    # Save metrics to a JSON file
    with open(f'{OUTPUTS_DIR}/metrics.json', 'w') as json_file:
        json.dump(all_metrics, json_file)
