import logging
from kombu import Producer

from core import Frame
from detection import Motion

logger = logging.getLogger('grab')


def grab_loop(config, stream, connection, exchange):
    logger.info('init queue producer')
    producer = Producer(connection)

    width, height = config['stream']['size']
    logger.info('stream size: {size}'.format(size=(width, height)))

    detector_sizes = [(int(width * sw), int(height * sh)) for sw, sh in config['stream']['cuts']]
    logger.info('detector sizes: {detector_sizes}'.format(detector_sizes=detector_sizes))

    logger.info('check init frame from stream')
    image = stream.read()

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
            logger.info('publish frame {bounds}'.format(bounds=motion_frame.bounds))
            producer.publish(motion_frame.to_json(),
                             exchange=exchange,
                             serializer='json', compression='zlib')

