from dataclasses import dataclass
from pathlib import Path

@dataclass
class Stamp:
    """
    Represents a stamp with a path and coordinates.

    Attributes:
    - path (Path): The path to the stamp image.
    - x1 (int): The x-coordinate of the upper-left corner.
    - y1 (int): The y-coordinate of the upper-left corner.
    - x2 (int): The x-coordinate of the lower-right corner.
    - y2 (int): The y-coordinate of the lower-right corner.
    """
    path: Path
    x1: int
    y1: int
    x2: int
    y2: int

    def __str__(self):
        """
        Returns a string representation of the stamp.

        Returns:
        str: A string with coordinates information.
        """
        return (
            f"Upper left corner coords (x, y): ({self.x1}, {self.y1})\n"
            f"Lower right corner coords (x, y): ({self.x2}, {self.y2})"
        )
