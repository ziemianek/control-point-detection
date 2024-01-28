import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor,
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights
)
from tqdm import tqdm

from src.common.utils import collate_fn, get_train_transform, get_valid_transform
from src.dataset import CustomDataset
from src.settings import (
    BATCH_SIZE,
    CLASSES,
    DEVICE,
    LEARNING_RATE,
    MOMENTUM,
    NUM_CLASSES,
    NUM_EPOCHS,
    OUTPUTS_DIR,
    WIDTH,
    HEIGHT,
    TRAIN_DIR,
    VALID_DIR,
    WEIGHT_DECAY
)


class Model:
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.train_loader = None
        self.valid_loader = None
        self.train_loss_history = []
        self.valid_loss_history = []
        self.precision_history = []
        self.recall_history = []

    def create_model(self, num_classes):
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
        self.criterion = torch.nn.CrossEntropyLoss()

    def create_data_loaders(self):
        train_dataset = CustomDataset(TRAIN_DIR, WIDTH, HEIGHT, CLASSES, get_train_transform())
        valid_dataset = CustomDataset(VALID_DIR, WIDTH, HEIGHT, CLASSES, get_valid_transform())

        # for testing purposes, to be deleted
        from torch.utils.data import Subset
        part = 1
        train_dataset = Subset(train_dataset, range(int(len(train_dataset)*part)))
        valid_dataset = Subset(valid_dataset, range(int(len(valid_dataset)*part)))

        self.train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn)
        self.valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn)

    def train_one_epoch(self):
        print("Training...")

        self.model.train()
        running_loss = 0.0

        for images, targets in tqdm(self.train_loader, total=len(self.train_loader)):
            images = [image.to(DEVICE) for image in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            self.optimizer.zero_grad()
            output = self.model(images, targets)
            loss = sum(loss for loss in output.values())
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            self.train_loss_history.append(loss.item())

        return running_loss / len(self.train_loader)

    def evaluate_one_epoch(self):
        print("Evaluating...")

        self.model.eval()

        avg_precisions = {}
        for images, targets in tqdm(self.valid_loader, total=len(self.valid_loader)):
            images = [image.to(DEVICE) for image in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            with torch.no_grad():
                output = self.model(images)

            for treshold in np.arange(0.5, 1, 0.05):
                treshold = round(treshold, 2)
                # AP for each batch
                AP = average_precision(
                    output,
                    targets,
                    iou_threshold=treshold
                )
                if not treshold in avg_precisions.keys():
                    avg_precisions[treshold] = []
                avg_precisions[treshold].append(AP)
        
        # epoch ap for every iou 
        for th, ap in avg_precisions.items():
            print(f"AP@IOU={th}: {sum(ap)/len(ap):.6f}")           

        # mAp for each epoch
        items_count = 0
        ap_sum = 0
        for th, ap in avg_precisions.items():
            ap_sum += sum(ap)
            items_count += len(ap)

        mAP = ap_sum / items_count

        return mAP

    def train(self, num_epochs):

        for epoch in range(num_epochs):
            print(f"Epoch [{epoch+1}/{num_epochs}]")

            train_loss = self.train_one_epoch()
            print(f"Train loss: {train_loss:.6f}")

            mAP = self.evaluate_one_epoch()
            print(f"mAP@.5:0.05:0.95: {mAP:.6f}")

            if epoch % 2 == 0:
                print(f"Saving model to {OUTPUTS_DIR} after {epoch+1} epochs")
                torch.save(self.model.state_dict(), f"{OUTPUTS_DIR}/model_{epoch+1}e.pth")
                print("Completed saving the model")

        print("Finished Training")

    def load_model(self, path):
        if not torch.cuda.is_available():
          checkpoint = torch.load(path, map_location=torch.device('cpu'))
        else:
          checkpoint = torch.load(path)

        self.model.load_state_dict(checkpoint)


if __name__ == "__main__":
    # Create an instance of FasterRCNNTrainer
    trainer = Model()

    # Create the model
    trainer.create_model(NUM_CLASSES)

    # # Create data loaders
    trainer.create_data_loaders()

    # # Train the model for the specified number of epochs
    trainer.train(NUM_EPOCHS)

    # # Save the trained model
    # torch.save(trainer.model.state_dict(), f"{OUTPUTS_DIR}/final_model.pth")
