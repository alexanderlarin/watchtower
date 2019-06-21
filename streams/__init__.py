import urllib.request

import numpy as np
import cv2.cv2 as cv2


class CgiStream:
    def __init__(self, url, user, password):
        password_manager = urllib.request.HTTPPasswordMgrWithDefaultRealm()
        password_manager.add_password(None, url, user, password)

        handler = urllib.request.HTTPBasicAuthHandler(password_manager)
        opener = urllib.request.build_opener(handler)

        self._stream = opener.open(url)
        self._bytes_read = b''

    def read(self):
        while True:
            self._bytes_read += self._stream.read(1024)
            a = self._bytes_read.find(b'\xff\xd8')
            b = self._bytes_read.find(b'\xff\xd9')
            if a != -1 and b != -1:
                image_data = self._bytes_read[a:b + 2]
                self._bytes_read = self._bytes_read[b + 2:]
                return cv2.imdecode(np.frombuffer(image_data, dtype=np.uint8), cv2.IMREAD_COLOR)

    def close(self):
        self._stream.close()


class Cv2Stream:
    def __init__(self, *args, **kwargs):
        self._stream = cv2.VideoCapture(*args, **kwargs)

    def read(self):
        is_ok, image = self._stream.read()
        return None if not is_ok else image

    def close(self):
        self._stream.release()
