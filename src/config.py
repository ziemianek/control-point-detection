# Config

import torch


# Paths
PROJECT_DIR_PATH = "/Users/ziemian/Code/bt"      # for local development
# PROJECT_DIR_PATH = "/content/Inzynierka"       # for google colab development
DATA_DIR_PATH = f"{PROJECT_DIR_PATH}/data"
OUTPUTS_DIR = f'{PROJECT_DIR_PATH}/outputs'      # dir to save model and plots
TEMPLATES_DIR_PATH = f"{PROJECT_DIR_PATH}/templates"

PHOTOS_DIR_PATH = f"{DATA_DIR_PATH}/photos"
POSITIVES_DIR_PATH = f"{DATA_DIR_PATH}/positives"
NEGATIVES_DIR_PATH = f"{DATA_DIR_PATH}/negatives"
ANNOTATIONS_DIR_PATH = f"{DATA_DIR_PATH}/annotations"

ANNOTATION_TEMPLATE_FILE_PATH = f"{TEMPLATES_DIR_PATH}/annotation_template.xml"
OBJECT_TEMPLATE_FILE_PATH = f"{TEMPLATES_DIR_PATH}/object_template.xml"

TRAIN_DIR = f'{DATA_DIR_PATH}/train'
TEST_DIR = f'{DATA_DIR_PATH}/test'
VALID_DIR = f'{DATA_DIR_PATH}/valid'

# Network configuration
BATCH_SIZE = 4         # increase / decrease according to GPU memeory
WIDTH = HEIGHT = 2048  # [px] resize the image for training and transforms
NUM_EPOCHS = 50        # number of epochs
LEARNING_RATE = 0.001  # https://deepchecks.com/glossary/learning-rate-in-machine-learning/
MOMENTUM = None        # https://datascience.stackexchange.com/questions/84167/what-is-momentum-in-neural-network
WEIGHT_DECAY = 0.0005  # https://machinelearningmastery.com/how-to-reduce-overfitting-in-deep-learning-with-weight-regularization/
IOU_THRESHOLD = 0.5    # min iou
PATIENCE = 3           # (For early stopping) number of epochs to wait for improvement
CLASSES = ['__background__', 'Stamp']
NUM_CLASSES = len(CLASSES)

# For NVIDIA GPUs
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# FIXME: For Apple M1/M2 (not working for M2 Pro)
# DEVICE = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
