import logging
import cv2
import numpy as np

logger = logging.getLogger('detection')


class Motion:
    def __init__(self, image_size, background_alpha, foreground_alpha, threshold, blur_size, min_area, compress_size=None):
        assert 0. < background_alpha < 1.
        assert 0. < foreground_alpha < 1.

        self._image_size = image_size
        self._background_alpha = background_alpha
        self._foreground_alpha = foreground_alpha
        self._threshold = threshold
        self._blur_size = blur_size
        self._min_area = min_area
        self._compress_size = tuple(compress_size)
        self._uncompress_scale = None if self._compress_size is None else \
            (self._image_size[0] / self._compress_size[0], self._image_size[1] / self._compress_size[1])

        self._background = None
        self._foreground = None

    def compress(self, image):
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return grayscale_image if self._compress_size is None else cv2.resize(grayscale_image, self._compress_size)

    def uncompress_rect(self, rect):
        return rect if self._uncompress_scale is None else ((int(rect[0] * self._uncompress_scale[0]),
                                                            int(rect[1] * self._uncompress_scale[1]),
                                                            int(rect[2] * self._uncompress_scale[0]),
                                                            int(rect[3] * self._uncompress_scale[1])))

    def reset(self, background, foreground):
        assert background.shape[1::-1] == self._image_size
        assert foreground.shape[1::-1] == self._image_size

        self._background = self.compress(background)
        self._foreground = self.compress(foreground)

    def detect(self, image):
        assert self._background is not None
        assert self._foreground is not None

        height, width = image.shape[:2]
        assert (width, height) == self._image_size

        image = self.compress(image)
        self._background = self._background_alpha * self._background + (1. - self._background_alpha) * image
        self._foreground = self._foreground_alpha * self._foreground + (1. - self._foreground_alpha) * image

        difference_image = cv2.absdiff(self._background, self._foreground).astype(np.uint8)

        difference_image = cv2.medianBlur(difference_image, self._blur_size)
        _, difference_image = cv2.threshold(difference_image, self._threshold,
                                            np.iinfo(difference_image.dtype).max, cv2.THRESH_BINARY)

        _, contours, _ = cv2.findContours(difference_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return [self.uncompress_rect(cv2.boundingRect(contour))
                for contour in contours if cv2.contourArea(contour) >= self._min_area]


class Face:
    def __init__(self, prototxt_filename, model_filename, blob_size, mean, threshold, scale_factor=1.):
        assert 0. < threshold <= 1.

        self._network = cv2.dnn.readNetFromCaffe(prototxt_filename, model_filename)
        self._blob_size = tuple(blob_size)
        self._mean = mean
        self._threshold = threshold
        self._scale_factor = scale_factor

    def detect(self, image):
        height, width = image.shape[:2]

        blob = cv2.dnn.blobFromImage(image, self._scale_factor, self._blob_size, self._mean)
        self._network.setInput(blob)

        detections = self._network.forward()

        rects = []
        for i in range(0, detections.shape[2]):
            # filter out weak detections by ensuring the predicted
            # probability is greater than a minimum threshold
            if detections[0, 0, i, 2] > self._threshold:
                # compute the (x, y)-coordinates of the bounding box for
                # the object, then update the bounding box rectangles list
                x, y, end_x, end_y = detections[0, 0, i, 3:7] * np.array([width, height, width, height])

                # cut by image size
                # TODO: use numpy operations
                x = max(x, 0.)
                y = max(y, 0.)
                end_x = max(x, min(end_x, float(width - 1)))
                end_y = max(y, min(end_y, float(height - 1)))

                w = end_x - x
                h = end_y - y
                if w != 0 and h != 0:
                    rects.append((int(x), int(y), int(w), int(h)))

        return rects
