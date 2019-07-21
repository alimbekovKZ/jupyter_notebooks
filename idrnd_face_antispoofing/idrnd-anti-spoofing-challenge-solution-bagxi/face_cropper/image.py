"""image loader"""

import io

import numpy as np
import PIL


class Image(np.ndarray):
    """Image is a wrapper around np.ndarray"""

    def __new__(cls, input_data):
        data = None
        if isinstance(input_data, Image):
            return input_data
        elif isinstance(input_data, str):
            data = PIL.Image.open(input_data)
        elif isinstance(input_data, np.ndarray):
            data = input_data
        elif isinstance(input_data, bytes):
            data = PIL.Image.open(io.BytesIO(input_data))
        else:
            raise TypeError('unsupported type')

        obj = np.array(data, dtype=np.uint8).view(cls)
        return obj

    def as_pil_image(self):
        """Pil Image representation of image
        Returns:
            Image: Returns Pil Image object
        """
        return PIL.Image.fromarray(self)

    def as_bytes(self, quality=100, img_format='jpeg'):
        """Bytes representation of image, as it will be read from file.
        Args:
            quality int: Quality of image. Default is 100.
            img_format string: Format of image. Default is jpeg.
        Returns:
            Image: Returns np array like object.

        """
        if img_format == 'jpg':
            _img_format = 'jpeg'
        else:
            _img_format = img_format
        img = PIL.Image.fromarray(self)
        img_bytes = io.BytesIO()
        img.save(img_bytes, format=_img_format, quality=quality)
        return img_bytes.getvalue()
