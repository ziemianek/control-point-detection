from src.config import *
from src.nnet.net import Net
from src.nnet.train import train
from src.nnet.eval import eval


if __name__ == "__main__":
    net = Net()

    print("Creating the model...")
    net.create_model(NUM_CLASSES)

    print("Creating data loaders...")
    net.create_data_loaders()

    print("Training the model...")
    train(net)

    print("Evaluating the model...")
    eval(net)
