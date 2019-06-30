import numpy as np


class Frame:
    def __init__(self, image, x, y, w, h):
        self._image = image
        self._x = x
        self._y = y
        self._w = w
        self._h = h

    @property
    def image(self):
        return self._image

    @property
    def position(self):
        return self._x, self._y

    @property
    def size(self):
        return self._w, self._h

    @property
    def bounds(self):
        return self._x, self._y, self._w, self._h

    def cut(self, x, y, w, h):
        return Frame(self._image[y:y + h, x: x + w, :], x + self._x, y + self._y, w, h)

    def to_json(self):
        return {
            'image': self._image.tobytes().decode('latin-1'),
            'x': self._x, 'y': self._y, 'w': self._w, 'h': self._h
        }

    @classmethod
    def from_json(cls, data):
        w = data['w']
        h = data['h']
        return cls(np.frombuffer(data['image'].encode('latin-1'), dtype=np.uint8).reshape([h, w, 3]),
                   data['x'], data['y'], w, h)
