import logging
from kombu import Producer

from core import Frame
from detection import Motion

logger = logging.getLogger('grab')


def grab_loop(config, stream, channel, exchange):
    logger.info('init queue producer')
    producer = Producer(channel)

    width, height = config['stream']['size']
    logger.info('stream size: {size}'.format(size=(width, height)))

    blob_size = tuple(config['face_detector']['blob_size'])
    logger.info('min grab size: {blob_size}'.format(blob_size=blob_size))

    logger.info('check init frame from stream')
    image = stream.read()

    motion_detector_config = config['motion_detector']
    logger.info('init motion detector: {motion_detector_config}'
                .format(motion_detector_config=motion_detector_config))
    motion = Motion(image_size=(width, height), **motion_detector_config)

    logger.info('reset motion detector')
    motion.reset(image, image)

    while True:
        image = stream.read()
        if image is None:
            logger.info('stream end')
            break

        frame = Frame(image, 0, 0, width, height)
        rects = motion.detect(frame.image)

        if len(rects):
            # TODO: use biggest rect, not a first
            # TODO: detector sizes is redundant feature after face detector became single
            # TODO: fit_cut should be greater than blob size but less than entire image size
            motion_frame = frame.cut(*rects[0])
            logger.info('publish frame {bounds}'.format(bounds=motion_frame.bounds))
            producer.publish(motion_frame.to_json(),
                             exchange=exchange,
                             serializer='json', compression='zlib')

