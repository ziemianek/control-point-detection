import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor,
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights
)
from src.common.utils import collate_fn, get_train_transform, get_valid_transform
from src.config import (
    BATCH_SIZE,
    CLASSES,
    DEVICE,
    LEARNING_RATE,
    MOMENTUM,
    WIDTH,
    HEIGHT,
    TRAIN_DIR,
    VALID_DIR,
    WEIGHT_DECAY
)
from src.processing.dataset import CustomDataset


class Net:
    def __init__(self) -> None:
        """
        Initializes an instance of the Net class.
        """
        self.model = None
        self.optimizer = None
        self.train_loader = None
        self.valid_loader = None


    def create_model(self, num_classes: int) -> None:
        """
        Creates the Faster R-CNN model with a custom head for the given number of classes.

        Parameters:
            num_classes (int): Number of classes in the dataset.

        Returns:
            None
        """
        model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, num_classes)
        model.to(DEVICE)

        self.model = model
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=LEARNING_RATE,
            momentum=MOMENTUM,
            weight_decay=WEIGHT_DECAY
        )


    def create_data_loaders(self) -> None:
        """
        Creates data loaders for training and validation datasets.

        Returns:
            None
        """
        train_dataset = CustomDataset(TRAIN_DIR, WIDTH, HEIGHT, CLASSES, get_train_transform())
        valid_dataset = CustomDataset(VALID_DIR, WIDTH, HEIGHT, CLASSES, get_valid_transform())
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn
        )
        self.valid_loader = DataLoader(
            valid_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn
        )
