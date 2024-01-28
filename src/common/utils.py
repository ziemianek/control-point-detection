import albumentations as A
import os
import shutil
import sys
from albumentations.pytorch import ToTensorV2


DEFAULT_ENCODING = sys.getdefaultencoding()


def collate_fn(batch: list) -> tuple:
    """
    Collate function to handle data loading with varying number of objects and tensor sizes.

    Parameters:
        batch (list): List of tuples containing batch data.

    Returns:
        tuple: Tuple of lists, each containing data for a specific aspect of the batch.
    """
    return tuple(zip(*batch))


def get_train_transform() -> A.Compose:
    """
    Returns the Albumentations composition for training data.
    """
    return A.Compose(
        [
            A.Flip(0.5),
            A.RandomRotate90(0.5),
            A.MotionBlur(p=0.5),
            A.MedianBlur(blur_limit=3, p=0.5),
            A.Blur(blur_limit=3, p=0.5),
            A.ChannelShuffle(p=0.5),
            ToTensorV2(p=1.0),
        ], 
        bbox_params={
            'format': 'pascal_voc',
            'label_fields': ['labels']
        }
    )

def get_valid_transform() -> A.Compose:
    """
    Returns the Albumentations composition for validation data.
    """
    return A.Compose(
        [
            ToTensorV2(p=1.0),
        ], 
        bbox_params={
            'format': 'pascal_voc', 
            'label_fields': ['labels']
        }
    )


def list_files_recursively(start_directory: str, filter: str = None) -> list:
    """
    Recursively lists files in a directory based on the given filter.

    Parameters:
        start_directory (str): The starting directory for listing files.
        filter (str, optional): File extension filter. Defaults to None.

    Returns:
        list: List of file paths.
    """
    listed_files = []

    if filter is not None:
        file_suffix = '.' + filter
    else:
        file_suffix = ''

    for root, dirs, files in os.walk(start_directory):
        for file in files:
            file_path = os.path.join(root, file)
            if file_path.endswith(file_suffix):
                listed_files.append(file_path)
    
    return listed_files


def read_file_content(file_path: str) -> list:
    """
    Reads the content of a file and returns it as a list of lines.

    Parameters:
        file_path (str): Path to the file.

    Returns:
        list: List of lines from the file.
    """
    try:
        with open(file_path, 'r', encoding=DEFAULT_ENCODING) as file:
            return file.readlines()
    except FileNotFoundError:
        print(f'{file_path} not found!')


def write_file(file_path: str, content: str) -> None:
    """
    Writes content to a file.

    Parameters:
        file_path (str): Path to the file.
        content (str): Content to be written to the file.

    Returns:
        None
    """
    try:
        with open(file_path, 'w') as file:
            file.write(content)
    except Exception:
        print(f'Couldn\'t write to file {file_path}')


def copy_file(src: str, dst: str) -> None:
    """
    Copies a file from source to destination.

    Parameters:
        src (str): Path to the source file.
        dst (str): Path to the destination file.

    Returns:
        None
    """
    if not os.path.exists(src):
        print(f"Directory '{src}' does not exist.")

    try:
        shutil.copy(src, dst)
        print(f"Copied {src} to {dst}")
    except Exception as e:
        print(f"Error copying {src} to {dst}:\n{str(e)}")


def rename_file(old_name: str, new_name: str) -> None:
    """
    Renames a file.

    Parameters:
        old_name (str): Current name of the file.
        new_name (str): New name for the file.

    Returns:
        None
    """
    try:
        if os.path.exists(new_name):
            raise FileExistsError(
                f"File '{new_name}' already exists in the destination directory."
            )

        os.rename(old_name, new_name)
        print(f"Renamed {old_name} to {new_name}")

    except FileExistsError as e:
        print(e)

    except Exception as e:
        print(f"Error renaming {old_name} to {new_name}:\n{str(e)}")


def remove_file(path: str) -> None:
    """
    Removes a file.

    Parameters:
        path (str): Path to the file.

    Returns:
        None
    """
    if os.path.isfile(path):
        try:
            os.remove(path)
            print(f"Successfully removed file: {path}")
        except Exception as e:
            print(f"Error removing file: {path}: {str(e)}")


def clear_directory(path: str) -> None:
    """
    Clears all files in a directory.

    Parameters:
        path (str): Path to the directory.

    Returns:
        None
    """
    files = os.listdir(path)
    for file in files:
        file_path = os.path.join(path, file)
        remove_file(file_path)

    print(f"All files in the {path} have been removed.")


def format_bboxes(bboxes: list) -> list:
    """
    Formats bounding boxes from the input format to a standardized format.

    Parameters:
        bboxes (list): List of dictionaries representing bounding boxes.

    Returns:
        list: List of formatted bounding boxes.
    """
    formatted_bboxes = []
    for i, p in enumerate(bboxes):
        for j in range(len(p['boxes'])):
            box = p['boxes'][j].numpy()
            label = int(p['labels'][j].numpy())
            if 'scores' in p:
                score = round(float(p['scores'][j].numpy()), 4)
                new_item = [i, label, score, box[0], box[1], box[2], box[3]]
            else:
                new_item = [i, label, box[0], box[1], box[2], box[3]]
            formatted_bboxes.append(new_item)
    return formatted_bboxes
