import argparse
import json
import logging.config
from kombu import Connection, Exchange, Queue

from grab import grab_loop
from streams import CgiStream, Cv2Stream
from watch import watch_loop


logger = logging.getLogger('tower')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Motion detection, people tracking and counting, '
                                                 'face, age and gender recognition service',
                                     epilog='Let the tool do the work!', )
    parser.add_argument('--config', default='config.json',
                        help='the path to JSON-formatted configuration file')
    parser.add_argument('--broker-url', default='redis://localhost:6379/',
                        help='the connection url for MQ broker')

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

    watch_parser = sub_parsers.add_parser('watch')

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
        video_stream = Cv2Stream(args.file or int(args.camera))
    elif args.stream == 'cgi':
        logger.info('init cgi video stream')
        video_stream = CgiStream(args.url, args.username, args.password)
    else:
        video_stream = None

    logger.info('connect to {broker_url}'.format(broker_url=args.broker_url))

    #: By default messages sent to exchanges are persistent (delivery_mode=2),
    #: and queues and exchanges are durable.
    exchange = Exchange('watch_tower', type='direct')
    queue = Queue(exchange=exchange, exclusive=True)

    with Connection(args.broker_url) as connection:
        if video_stream is not None:
            with connection.channel() as channel:
                grab_loop(config, video_stream, channel, exchange)
        else:
            watch_loop(config, connection, queue)
