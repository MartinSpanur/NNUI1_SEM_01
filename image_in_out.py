from pathlib import Path

import numpy as np
from PIL import Image


def bmp_to_binary_matrix(
    image_path: str | Path,
    white_threshold: int = 245,
) -> np.ndarray:
    """
    Convert a BMP image to a binary NumPy matrix.

    Pixels whose red, green, and blue channels are all greater than or
    equal to `white_threshold` are treated as white and represented by 0.
    All other pixels are represented by 1.

    Args:
        image_path:
            Path to the BMP image.

        white_threshold:
            Minimum RGB channel value for a pixel to be considered
            white or close to white. Must be between 0 and 255.

            Examples:
                255 -> only pure white becomes 0
                245 -> white and very light colors become 0
                230 -> a broader range of light colors becomes 0

    Returns:
        A 2D NumPy array with dtype uint8 containing only 0 and 1.

    Raises:
        ValueError:
            If white_threshold is outside the valid range.
    """

    if not 0 <= white_threshold <= 255:
        raise ValueError("white_threshold must be between 0 and 255.")

    with Image.open(image_path) as image:
        rgb_image = np.asarray(image.convert("RGB"))

    # A pixel is near-white only when all three RGB channels are bright.
    near_white = np.all(rgb_image >= white_threshold, axis=2)

    # Near-white -> 0, everything else -> 1
    return (~near_white).astype(np.uint8)



def matrix_to_bmp(
    matrix: np.ndarray,
    output_path: str | Path,
) -> None:
    """
    Convert a 2D NumPy matrix into a BMP image.

    Color mapping:
        0 -> white   (255, 255, 255)
        1 -> black   (0, 0, 0)
        2 -> cyan    (0, 255, 255)
        3 -> magenta (255, 0, 255)
        4 -> yellow  (255, 255, 0)

    Args:
        matrix:
            A two-dimensional NumPy array containing values from 0 to 4.

        output_path:
            Path where the BMP image will be saved.

    Raises:
        ValueError:
            If the matrix is not two-dimensional or contains unsupported
            values.
    """
    matrix = np.asarray(matrix)

    if matrix.ndim != 2:
        raise ValueError(
            f"Expected a 2D matrix, but received shape {matrix.shape}."
        )

    if not np.all(np.isin(matrix, [0, 1, 2, 3, 4])):
        invalid_values = np.unique(
            matrix[~np.isin(matrix, [0, 1, 2, 3, 4])]
        )
        raise ValueError(
            f"Matrix contains unsupported values: {invalid_values}. "
            "Allowed values are 0, 1, 2, 3, and 4."
        )

    # Each row index corresponds to the matrix value.
    color_map = np.array(
        [
            [255, 255, 255],  # 0: white
            [0, 0, 0],        # 1: black
            [0, 255, 255],    # 2: cyan
            [255, 0, 255],    # 3: magenta
            [255, 255, 0],    # 4: yellow
        ],
        dtype=np.uint8,
    )

    rgb_pixels = color_map[matrix.astype(np.uint8)]
    image = Image.fromarray(rgb_pixels, mode="RGB")
    image.save(output_path, format="BMP")