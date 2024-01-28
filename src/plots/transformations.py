import albumentations as A
import cv2
import os
import glob



# Sample image path
image_path = "/Users/ziemian/Code/bt/paper/images/fotopunkt.png"
output_directory = "/Users/ziemian/Code/bt/paper"


def apply_and_save_transformations(image_path, output_dir):
    # Load the image
    image = cv2.imread(image_path)

    # Define transformations
    transformations = {
        "flip": A.Flip(always_apply=True, p=1.0),
        "randomRotate90": A.RandomRotate90(always_apply=True, p=1.0),
        "medianBlur3x3": A.MedianBlur(blur_limit=7, always_apply=True, p=1.0),
        "channelShuffle": A.ChannelShuffle(always_apply=True, p=1.0),
        "colorJitter": A.ColorJitter(always_apply=True, p=1.0)
    }

    # Apply each transformation and save the result
    for t_name, transform in transformations.items():
        transformed_image = transform(image=image)["image"]
        output_path = os.path.join(output_dir, f"transform_{t_name}.png")
        cv2.imwrite(output_path, transformed_image)


def visualize(image, targets):
    if len(targets) == 0:
      return

    for i, b in enumerate(targets):
      id, label, *box = b
      cv2.rectangle(image,
                  (int(box[0]), int(box[1])),
                  (int(box[2]), int(box[3])),
                  (0, 0, 255), 1)
      cv2.putText(image, f'Target {i+1}/{len(targets)}',
                  (int(box[2]), int(box[3]+5)),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                  2, lineType=cv2.LINE_AA)
    cv2.imwrite(f"{output_directory}/{image_name}_transformation.png", image)

if __name__ == "__main__":
    # directory where all the images are present
    test_images = glob.glob(f"{output_directory}/../data/test/*")
    import xml.etree.ElementTree as ET
    import os
    for root, dirs, files in os.walk(f"{output_directory}/../data/annotations"):
        image_annotations = [f.split('.')[0] for f in files]

    print(f"Test instances: {len(test_images)}")

    for i in range(len(test_images) // 4):
        # get the image file name for saving output later on
        image_name = test_images[i].split('/')[-1].split('.')[0]


        for j in image_annotations:
            if j == image_name:
            annotated_boxes = []
            tree = ET.parse(f'{ANNOTATIONS_DIR_PATH}/{j}.xml')
            root = tree.getroot()

            # box coordinates for xml files are extracted and corrected for image size given
            for member in root.findall('object'):
                # xmin = left corner x-coordinates
                xmin = int(member.find('bndbox').find('xmin').text)
                # xmax = right corner x-coordinates
                xmax = int(member.find('bndbox').find('xmax').text)
                # ymin = left corner y-coordinates
                ymin = int(member.find('bndbox').find('ymin').text)
                # ymax = right corner y-coordinates
                ymax = int(member.find('bndbox').find('ymax').text)

                annotated_boxes.append([xmin, ymin, xmax, ymax])

        image = cv2.imread(test_images[i])
        orig_image = image.copy()

        # BGR to RGB
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)

        # make the pixel range between 0 and 1
        image /= 255.0

        # bring color channels to front
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        # convert to tensor
        image = torch.tensor(image, dtype=torch.float).cpu()

        # add batch dimension
        image = torch.unsqueeze(image, 0)


        for j in range(len(annotated_boxes)):
            annotated_boxes[j].insert(0, 0)
            annotated_boxes[j].insert(1, 1)


    visualize(orig_image, annotated_boxes)
    # Output directory for transformed images
    apply_and_save_transformations(image_path, output_directory)
