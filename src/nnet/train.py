import time
import torch
import tqdm
from src.config import (
    OUTPUTS_DIR,
    NUM_EPOCHS,
    DEVICE,
    PATIENCE,
    TRAIN_LOSS_FILE_PATH,
    VALID_LOSS_FILE_PATH
)
from src.common.utils import append_file
from src.nnet.net import Net


def train(net: Net) -> None:
    """
    Training loop for the neural network.

    Parameters:
        net (Net): Neural network model.

    Returns:
        None
    """
    train_losses = []
    valid_losses = []
    best_loss = float('inf')
    current_patience = 0

    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        train_loss = 0
        valid_loss = 0

        # train
        net.model.train()
        for images, targets in tqdm.tqdm(net.train_loader, total=len(net.train_loader)):
            images = [image.to(DEVICE) for image in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            net.optimizer.zero_grad()

            output = net.model(images, targets)

            loss = sum(l for l in output.values())
            train_loss += loss.item()

            loss.backward()
            net.optimizer.step()

        train_loss /= len(net.train_loader)
        append_file(TRAIN_LOSS_FILE_PATH, f"epoch: {epoch + 1}, train_loss: {train_loss}")
        train_losses.append(train_loss)
        print(f"Train loss for epoch {epoch + 1}: {train_loss}")

        # valid
        for images, targets in tqdm.tqdm(net.valid_loader, total=len(net.valid_loader)):
            images = [image.to(DEVICE) for image in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            with torch.no_grad():
                output = net.model(images, targets)

            loss = sum(l for l in output.values())
            valid_loss += loss.item()

        valid_loss /= len(net.valid_loader)  # sum of loss for batch / num batches
        append_file(VALID_LOSS_FILE_PATH, f"epoch: {epoch + 1}, valid_loss: {valid_loss}")
        valid_losses.append(valid_loss)
        print(f"Validation loss for epoch {epoch + 1}: {valid_loss}")

        print(f"Saving model to {OUTPUTS_DIR} after {epoch + 1} epochs")
        torch.save(net.model.state_dict(), f"{OUTPUTS_DIR}/model_{epoch + 1}e.pth")
        print("Completed saving the model")

        # === Start if early stopping mechanism ===
        # Check for improvement in validation loss
        if valid_loss < best_loss:
            best_loss = valid_loss
            current_patience = 0
        else:
            current_patience += 1

        # Check if early stopping criteria are met
        if current_patience >= PATIENCE:
            print(f'Early stopping after {epoch + 1} epochs without improvement.')
            break
        # === End of early stopping mechanism ===

    total_time = (time.time() - start_time) / 60
    print(f"Training took: {round(total_time, 2)} minutes")
