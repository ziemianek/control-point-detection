import matplotlib.pyplot as plt
import os

dir = "/Users/ziemian/Code/bt/paper"
train_loss_filepath = f"{dir}/train_loss.txt"
valid_loss_filepath = f"{dir}/val_loss.txt"
loss_fig = "loss_plot.png"
fig_size = (14,8)

def read_loss_values(file):
    try:
        with open(file, "r") as f:
            return f.readlines()
    except:
        print(f"Couldn't read {file} content")


def parse_loss_values(content):
    loss = []
    for line in content:
        loss.append(line.split()[-1])
    return list(float(l) for l in loss)


if __name__ == "__main__":
    train_loss = parse_loss_values(read_loss_values(train_loss_filepath))
    valid_loss = parse_loss_values(read_loss_values(valid_loss_filepath))

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
    plt.savefig(f'{dir}/{loss_fig}')
