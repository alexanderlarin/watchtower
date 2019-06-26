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
        return Frame(self._image[y:y + h, x: x + w], x + self._x, y + self._y, w, h)

    def fit_cut(self, x, y, w, h, cut_sizes):
        f_w, f_h = next(((s_w, s_h) for s_w, s_h in cut_sizes if w <= s_w and h <= s_w), (w, h))

        f_w = min(f_w, self._w)
        f_h = min(f_h, self._h)

        x = int(x + (w - f_w) / 2.)
        y = int(y + (h - f_h) / 2.)

        x = max(self._x, x + min(self._w - (x + f_w), 0))
        y = max(self._y, y + min(self._h - (y + f_h), 0))
        w = f_w
        h = f_h

        return self.cut(x, y, w, h)

    def to_json(self):
        return {
            'image': self._image.tolist(),
            'x': self._x, 'y': self._y, 'w': self._w, 'h': self._h
        }

    @classmethod
    def from_json(cls, data):
        return cls(np.array(data['image'], dtype=np.uint8), data['x'], data['y'], data['w'], data['h'])
