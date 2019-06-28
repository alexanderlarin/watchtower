import time


class Timer:
    def __init__(self):
        self.start = None
        self.end = None
        self.interval = None

    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start

    def __str__(self):
        return '{interval} ms'.format(interval=self.interval * 1000)
