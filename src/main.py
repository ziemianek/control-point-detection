from src.config import *
from src.nnet.net import Net
from src.nnet.train import train
from src.nnet.eval import eval


if __name__ == "__main__":
    net = Net()
    net.create_model()
    net.create_data_loaders()

    train(net)
    eval(net)
