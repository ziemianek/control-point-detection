from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from dataclasses import dataclass
from numpy import array_split
from src.config import PHOTOS_DIR_PATH
from src.processing.stamp import Stamp


@dataclass
class Photo:
    """
    Represents a photo with stamps and associated metadata.

    Args:
        photo_path (str): The path to the photo.
        stamp_data (list): List of stamp data.

    Attributes:
        _path (Path): The normalized path to the photo.
        _filename (str): The filename of the photo.
        _data (list): List of stamp data.
        _metadata (dict): Metadata extracted from the photo.
        _stamps (list): List of Stamp objects.

    Methods:
        path: Property to get the photo's path.
        filename: Property to get the filename of the photo.
        stamps: Property to get the list of Stamp objects.
        metadata: Property to get the extracted metadata.
        has_stamps(): Check if the photo has stamps.

    Private Methods:
        _has_positive_data(): Check if stamp data is present.
        _create_stamps(): Create Stamp objects from stamp data.
        _normalize_path(photo_path): Normalize the photo's path.
        _extract_filename(): Extract the filename from the path.
        _extract_metadata(): Extract metadata from the photo.
        _format_metadata(metadata): Format and filter relevant metadata.
        _dms_to_decimal(dms, direction): Convert degrees, minutes, seconds to decimal degrees.
    """

    def __init__(self, path, data=None):
        self._path = self._normalize_path(path)
        self.filename = self._extract_filename()
        self._data = data
        self._metadata = self._extract_metadata()

        if self._has_positive_data():
            self._stamps = self._create_stamps()
        else:
            self._stamps = None

    def __str__(self):
        """
        Returns a string representation of the photo.

        Returns:
            str: A string containing photo details and stamps.
        """
        return (
            f"Photo: {self._path}\n"
            f"Number of stamps: {len(self._stamps) if self._stamps else 0}\n"
            # f"Stamps: {[str(stamp) for stamp in self._stamps]}"
        ) if self._stamps is not None else f"{self._path} has no stamps"

    @property
    def path(self):
        """
        Property to get the photo's path.

        Returns:
            str: The normalized path to the photo.
        """
        return str(self._path)

    @path.setter
    def path(self, value):
        self._path = value
        return self._path

    # @property
    # def filename(self):
    #     """
    #     Property to get the filename of the photo.

    #     Returns:
    #         str: The filename of the photo.
    #     """
    #     return self._filename

    # @path.setter
    # def filename(self, value):
    #     self._filename = value
    #     return self._filename

    @property
    def stamps(self):
        """
        Property to get the list of Stamp objects.

        Returns:
            list: A list of Stamp objects.
        """
        return self._stamps

    @property
    def metadata(self):
        """
        Property to get the extracted metadata.

        Returns:
            dict: Metadata extracted from the photo.
        """
        return self._metadata

    def has_stamps(self):
        """
        Check if the photo has stamps.

        Returns:
            bool: True if the photo has stamps, otherwise False.
        """
        return self._stamps is not None

    def _has_positive_data(self):
        """
        Check if stamp data is present.

        Returns:
            bool: True if stamp data is present, otherwise False.
        """
        return self._data is not None

    def _create_stamps(self):
        """
        Create Stamp objects from stamp data.

        Returns:
            list: A list of Stamp objects.
        """
        stamps = []

        num_stamps = int(self._data[0])
        stamps_info = [int(d) for d in self._data[1:]]

        for stamp_info in array_split(stamps_info, num_stamps):
            x1, y1, width, height = stamp_info
            x2, y2 = x1 + width, y1 + height

            stamps.append(
                Stamp(
                    path=self._path,
                    x1=x1, y1=y1,
                    x2=x2, y2=y2
                )
            )

        return stamps

    def _normalize_path(self, photo_path):
        """
        Normalize the photo's path.

        Args:
            photo_path (str): The original path.

        Returns:
            Path: The normalized path as a Path object.
        """
        old_path_prefix = r'C:\Users\nohax\Documents\!michal\_ACHIM'
        return photo_path.replace(old_path_prefix, PHOTOS_DIR_PATH).replace('\\', '/')

    def _extract_filename(self):
        """
        Extract the filename from the path.

        Returns:
            str: The extracted and lowercased filename.
        """
        return self._path.split('/')[-1]

    def _extract_metadata(self):
        """
        Extract metadata from the photo.

        Returns:
            dict: A dictionary containing extracted metadata.
        """
        metadata = {}

        try:
            image = Image.open(self._path)
        except Exception as e:
            return None

        exif_data = image._getexif()
        if exif_data is not None:
            for tag, value in exif_data.items():
                tag_name = TAGS.get(tag, tag)
                if tag_name not in ["MakerNote", "XPComment", "XPKeywords"]:
                    metadata[tag_name] = value
                if tag_name == "GPSInfo":
                    gps_info = value
                    for key, sub_value in gps_info.items():
                        sub_tag = GPSTAGS.get(key, key)
                        metadata[sub_tag] = sub_value

        return self._format_metadata(metadata)

    def _format_metadata(self, metadata):
        """
        Format and filter relevant metadata.

        Args:
            metadata (dict): Metadata extracted from the photo.

        Returns:
            dict: Formatted and filtered metadata.
        """
        relevant_keys = [
            "ExifImageWidth", "ExifImageHeight",
            "GPSLatitude", 'GPSLatitudeRef',
            'GPSLongitude', 'GPSLongitudeRef',
            'GPSAltitude'
        ]
        formatted_metadata = {key: metadata[key] for key in metadata if key in relevant_keys}

        formatted_metadata['GPSLatitude'] = self._dms_to_decimal(
            formatted_metadata['GPSLatitude'],
            formatted_metadata['GPSLatitudeRef']
        )

        formatted_metadata['GPSLongitude'] = self._dms_to_decimal(
            formatted_metadata['GPSLongitude'],
            formatted_metadata['GPSLongitudeRef']
        )

        return formatted_metadata

    def _dms_to_decimal(self, dms, direction):
        """
        Convert degrees, minutes, seconds to decimal degrees.

        Args:
            dms (tuple): Degrees, minutes, seconds.
            direction (str): The direction indicator (N/S for latitude, E/W for longitude).

        Returns:
            float: The converted decimal degrees.
        """
        degrees, minutes, seconds = dms
        decimal_degrees = degrees + (minutes / 60) + (seconds / 3600)

        if direction in ['S', 'W']:
            decimal_degrees = -decimal_degrees

        return float(decimal_degrees)
