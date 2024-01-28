import matplotlib.pyplot as plt
from src.config import TRAIN_LOSS_FILE_PATH, VALID_LOSS_FILE_PATH, OUTPUTS_DIR

loss_fig = "loss_plot.png"
fig_size = (14, 8)


def read_loss_values(file):
    """
    Read the content of a loss file.

    Parameters:
        file (str): Path to the loss file.

    Returns:
        List[str]: List containing lines from the loss file.
    """
    try:
        with open(file, "r") as f:
            return f.readlines()
    except Exception:
        print(f"Couldn't read {file} content")


def parse_loss_values(content):
    """
    Parse loss values from content.

    Parameters:
        content (List[str]): List containing lines of loss values.

    Returns:
        List[float]: List containing parsed loss values.
    """
    loss = []
    for line in content:
        loss.append(line.split()[-1])
    return list(float(l) for l in loss)


if __name__ == "__main__":
    train_loss = parse_loss_values(read_loss_values(TRAIN_LOSS_FILE_PATH))
    valid_loss = parse_loss_values(read_loss_values(VALID_LOSS_FILE_PATH))

    # Generate a list of epochs (assuming one epoch per entry)
    epochs = list(range(1, len(train_loss) + 1))

    plt.figure(figsize=fig_size)  # Adjust width and height as needed

    # Plot both train and validation loss values against epochs
    plt.plot(epochs, train_loss, label='Zbiór testowy', marker='o', linestyle='-')
    plt.plot(epochs, valid_loss, label='Zbiór walidacyjny', marker='o', linestyle='-')

    # Annotate each data point with its corresponding loss value
    for i, txt in enumerate(train_loss):
        plt.annotate(f'{txt:.4f}', (epochs[i], train_loss[i]), textcoords="offset points", xytext=(0, 10), ha='center')

    for i, txt in enumerate(valid_loss):
        plt.annotate(f'{txt:.4f}', (epochs[i], valid_loss[i]), textcoords="offset points", xytext=(0, -15), ha='center')

    plt.title('Wykres wartości funkcji straty dla zbiorów treningowego i walidacyjnego')
    plt.xlabel('Epoka')
    plt.ylabel('Wartości funkcji straty')
    plt.xticks(epochs)  # Set x-axis ticks to display every epoch
    plt.legend()  # Display legend
    plt.grid(True)
    # plt.show()
    plt.savefig(f'{OUTPUTS_DIR}/{loss_fig}')
