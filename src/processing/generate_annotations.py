import src.config as settings
from src.common.utils import (
    list_files_recursively,
    read_file_content,
    copy_file,
    rename_file,
    clear_directory,
    write_file
)
from src.processing.photo import Photo


class InvalidFormat(Exception):
    pass


class AnnotationGenerator:
    def __init__(self):
        self._photos_dir_path = settings.PHOTOS_DIR_PATH
        self._photos = []

    def _load_old_annotations(self):
        old_annotations_files = list_files_recursively(self._photos_dir_path, filter='txt')
        old_annotations = []

        for file in old_annotations_files:
            old_annotations += read_file_content(file)
        
        return old_annotations

    def _separete_positives_and_negatives(self):
        positives = []
        negatives = []

        for photo in self._photos:
            if photo.has_stamps():
                positives.append(photo)
            else:
                negatives.append(photo)

        return positives, negatives

    def _populate_photos(self, photo_data):
        for p in photo_data:
            p = p.split()

            # if true, then photo data has only path, thus doesn't contain annotated stamps
            if len(p) == 1:
                self._photos.append(Photo(path=p[0]))
            else:
                self._photos.append(Photo(path=p[0], data=p[1:]))

    def _create_PascalVOC_annotations(self, positives):
        clear_directory(settings.ANNOTATIONS_DIR_PATH)

        annotation_template = ''.join(read_file_content(settings.ANNOTATION_TEMPLATE_FILE_PATH))
        object_template = ''.join(read_file_content(settings.OBJECT_TEMPLATE_FILE_PATH))

        objects = list()

        for photo in positives:
            new_annotation = annotation_template
            new_annotation = new_annotation.replace('<folder></folder>', f'<folder>{settings.POSITIVES_DIR_PATH}</folder>')
            new_annotation = new_annotation.replace('<filename></filename>', f'<filename>{photo.filename}</filename>')
            new_annotation = new_annotation.replace('<path></path>', f'<path>{photo.path}</path>')
            new_annotation = new_annotation.replace('<width></width>', f'<width>{photo.metadata["ExifImageWidth"]}</width>')
            new_annotation = new_annotation.replace('<height></height>', f'<height>{photo.metadata["ExifImageHeight"]}</height>')

            for stamp in photo.stamps:
                new_object = object_template
                new_object = new_object.replace('<xmin></xmin>', f'<xmin>{stamp.x1}</xmin>')
                new_object = new_object.replace('<ymin></ymin>', f'<ymin>{stamp.y1}</ymin>')
                new_object = new_object.replace('<xmax></xmax>', f'<xmax>{stamp.x2}</xmax>')
                new_object = new_object.replace('<ymax></ymax>', f'<ymax>{stamp.y2}</ymax>')
                objects.append(new_object)

            object_replacement = '\n'.join(objects)
            objects.clear()

            new_annotation = new_annotation.replace('<object></object>', f'{object_replacement}')

            # {photo.filename.split(".")[0]}.xml ---> e.g. image name is 00001.jpg,
            # so we want annotation file to be named 00001.xml, thus we split image name
            # on `.` to get list of ['name', 'jpg']. Then i use the name (0th index) and .xml
            file = f'{settings.ANNOTATIONS_DIR_PATH}/{photo.filename.split(".")[0]}.xml'
            write_file(file, new_annotation)

    def _move_and_rename_photos(self, photos, dir):
        clear_directory(dir)
        for i, p in enumerate(photos):
            old_file_name = p.filename
            new_file_name = f'{i:05}.jpg'
            copy_file(p.path, dir)
            rename_file(
                old_name=f'{dir}/{old_file_name}',
                new_name=f'{dir}/{new_file_name}'
            )
            p.path = f'{dir}/{new_file_name}'
            p.filename = new_file_name

    def generate_annotations(self, format='pascal_voc'):
        old_annotations = self._load_old_annotations()
        self._populate_photos(old_annotations)
        positives, negatives = self._separete_positives_and_negatives()

        sum_stamps = 0
        for photo in positives:
            sum_stamps += len(photo.stamps)

        print(sum_stamps)
        # TODO: split those into separate methods to be able to move, rename or both
        # self._move_and_rename_photos(positives, settings.POSITIVES_DIR_PATH)

        # if format == 'pascal_voc':
        #     self._create_PascalVOC_annotations(positives)
        # else:
        #     raise InvalidFormat


if __name__ == "__main__":
    generator = AnnotationGenerator()
    generator.generate_annotations()
