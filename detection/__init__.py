import logging
import numpy as np
import cv2.cv2 as cv2

logger = logging.getLogger('detection')


class Motion:
    def __init__(self, background_alpha, foreground_alpha, threshold, blur_size, min_area, compress_size=None):
        assert 0. < background_alpha < 1.
        assert 0. < foreground_alpha < 1.

        self._background_alpha = background_alpha
        self._foreground_alpha = foreground_alpha
        self._threshold = threshold
        self._blur_size = blur_size
        self._min_area = min_area
        self._compress_size = compress_size
        self._uncompress_scale = None

        self._image_size = None
        self._background = None
        self._foreground = None

    def compress(self, image):
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return grayscale_image if self._compress_size is None else cv2.resize(grayscale_image,
                                                                              (self._compress_size[1], self._compress_size[0]))

    def uncompress_rect(self, rect):
        return rect if self._uncompress_scale is None else ((int(rect[0] * self._uncompress_scale[1]),
                                                            int(rect[1] * self._uncompress_scale[0]),
                                                            int(rect[2] * self._uncompress_scale[1]),
                                                            int(rect[3] * self._uncompress_scale[0])))

    def reset(self, background, foreground):
        assert background.shape == foreground.shape

        self._image_size = background.shape[:2]
        self._uncompress_scale = None if self._compress_size is None else \
            (self._image_size[0] / self._compress_size[0], self._image_size[1] / self._compress_size[1])

        self._background = self.compress(background)
        self._foreground = self.compress(foreground)

    def detect(self, image):
        assert np.all(image.shape[:2] == self._image_size)

        image = self.compress(image)
        self._background = self._background_alpha * self._background + (1. - self._background_alpha) * image
        self._foreground = self._foreground_alpha * self._foreground + (1. - self._foreground_alpha) * image

        difference_image = cv2.absdiff(self._background, self._foreground).astype(np.uint8)

        difference_image = cv2.medianBlur(difference_image, self._blur_size)
        _, difference_image = cv2.threshold(difference_image, self._threshold,
                                            np.iinfo(difference_image.dtype).max, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(difference_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return [self.uncompress_rect(cv2.boundingRect(contour))
                for contour in contours if cv2.contourArea(contour) >= self._min_area]
