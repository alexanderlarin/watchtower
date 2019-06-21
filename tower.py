import argparse
import json
import logging.config
import multiprocessing

from core import Frame
from detection import Motion
from streams import CgiStream, Cv2Stream
from watch import watch_loop


logger = logging.getLogger('tower')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Motion detection, people tracking and counting, '
                                                 'face, age and gender recognition service',
                                     epilog='Let the tool do the work!', )
    parser.add_argument('--config', default='config.json', help='the path to JSON-formatted configuration file')

    sub_parsers = parser.add_subparsers(title='Streams', dest='stream')

    # TODO: complete help and description
    # TODO: additional or and other checks
    url_parser = sub_parsers.add_parser('cgi')
    url_parser.add_argument('--url')
    url_parser.add_argument('--username')
    url_parser.add_argument('--password')

    cv2_parser = sub_parsers.add_parser('cv2')
    cv2_parser.add_argument('--camera')
    cv2_parser.add_argument('--file')

    args = parser.parse_args()

    # TODO: replace with constants somehow
    if args.stream is None:
        raise ValueError('stream parameter is required')

    # init logging
    with open(args.config) as config_file:
        config = json.load(config_file)
    logging.config.dictConfig(config['logging'])

    # stream = Cv2Stream('samples/100501m.avi')
    # video_stream = cv2.VideoCapture('videos/archive_542336805.avi')
    # video_stream = cv2.VideoCapture(0)
    # TODO: it's a fucking params hell
    if args.stream == 'cv2':
        logger.info('init cv2 video stream')
        stream = Cv2Stream(args.file or int(args.camera))
    elif args.stream == 'cgi':
        logger.info('init cgi video stream')
        stream = CgiStream(args.url, args.username, args.password)
    else:
        raise ValueError('stream parameter {arg} is not valid'.format(arg=args.stream))

    logger.info('init watch queue')
    watch_queue = multiprocessing.Queue()

    image = stream.read()

    height, width = image.shape[:2]
    detector_sizes = [(width // 4, height // 2), (width // 2, height), (width, height)]
    logger.info('detector sizes: {detector_sizes}'.format(detector_sizes=detector_sizes))

    logger.info('init watch process')
    watch_process = multiprocessing.Process(target=watch_loop, args=(config, watch_queue, detector_sizes,))

    logger.info('start watch process')
    watch_process.start()

    logger.info('init motion detector')
    motion = Motion(image_size=(width, height), **config['motion_detector'])

    logger.info('reset motion detector')
    motion.reset(image, image)

    while True:
        image = stream.read()
        if image is None:
            break

        frame = Frame(image, 0, 0, width, height)
        rects = motion.detect(frame.image)

        if len(rects):
            # TODO: use biggest rect, not a first
            motion_frame = frame.fit_cut(*rects[0], detector_sizes)
            watch_queue.put(motion_frame)

    stream.close()
    watch_queue.put(None)  # poison pill to end the queue
    logger.info('end video stream')

    watch_process.join()
    logger.info('end watch queue')
