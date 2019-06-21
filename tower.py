import argparse
import json
import logging.config
import multiprocessing
import os

import cv2.cv2 as cv2

from core import Frame
from detection import Motion
from watch import watch_loop


logger = logging.getLogger('tower')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Motion detection, people tracking and counting, '
                                                 'face, age and gender recognition service',
                                     epilog='Let the tool do the work!', )
    parser.add_argument('--config', default='config.json', help='the path to JSON-formatted configuration file')

    args = parser.parse_args()

    with open(args.config) as config_file:
        config = json.load(config_file)

    logging.config.dictConfig(config['logging'])

    logger.info('init watch queue')
    watch_queue = multiprocessing.Queue()

    logger.info('init video stream')
    video_stream = cv2.VideoCapture('samples/100501m.avi')
    # video_stream = cv2.VideoCapture('videos/archive_542336805.avi')
    # video_stream = cv2.VideoCapture(0)

    _, buff = video_stream.read()

    height, width = buff.shape[:2]
    detector_sizes = [(width // 4, height // 2), (width // 2, height), (width, height)]
    logger.info(f'detector sizes: {detector_sizes}')

    logger.info('init watch process')
    watch_process = multiprocessing.Process(target=watch_loop, args=(config, watch_queue, detector_sizes,))

    logger.info('start watch process')
    watch_process.start()

    logger.info('init motion detector')
    motion = Motion(image_size=(width, height), background_alpha=.9, foreground_alpha=.1,
                    threshold=10, blur_size=5, min_area=1000, compress_size=(320, 270))

    logger.info('reset motion detector')
    motion.reset(buff, buff)

    while True:
        _, image = video_stream.read()
        if image is None:
            break

        frame = Frame(image, 0, 0, width, height)
        rects = motion.detect(frame.image)

        if len(rects):
            # TODO: use biggest rect, not a first
            motion_frame = frame.fit_cut(*rects[0], detector_sizes)
            watch_queue.put(motion_frame)

    video_stream.release()
    watch_queue.put(None)  # poison pill
    logger.info('end video stream')

    watch_process.join()
    logger.info('end watch queue')
