import logging
import os
import uuid
import cv2
from kombu import Consumer, eventloop

from core import Frame
from detection import Face
from tracking import CentroidTracker


logger = logging.getLogger('watch')


def watch_loop(config, connection, queue):
    # TODO: args or config variable
    detections_dir = 'images/'
    if not os.path.exists(detections_dir) or not os.path.isdir(detections_dir):
        logger.warning('detections dir {detections_dir} is not exist, mkdir'
                       .format(detections_dir=detections_dir))
        os.makedirs(detections_dir)
    width, height = config['stream']['size']
    logger.info('stream size: {size}'.format(size=(width, height)))

    detector_sizes = [(int(width * sw), int(height * sh)) for sw, sh in config['stream']['cuts']]
    logger.info('detector sizes: {detector_sizes}'.format(detector_sizes=detector_sizes))

    logger.info('init face detectors: {detector_sizes_count}'.format(detector_sizes_count=len(detector_sizes)))
    face_detectors = [Face(size, **config['face_detector']) for size in detector_sizes]

    logger.info('init centroid tacker')
    centroid_tracker = CentroidTracker(**config['centroid_tracker'])

    def handle_message(body, message):
        frame = Frame.from_json(body)
        message.ack()

        face_detector = next((d for d in face_detectors if d.image_size == frame.size), None)
        face_rects = face_detector.detect(frame.image)
        frame_x, frame_y = frame.position
        face_origin_rects = [(frame_x + x, frame_y + y, w, h) for x, y, w, h in face_rects]

        logger.info('update centroid tacker frame: {frame_size}, faces: {face_rects_count}'
                    .format(frame_size=frame.size, face_rects_count=len(face_rects)))
        objs = centroid_tracker.update(face_origin_rects)

        draw_image = frame.image.copy()
        for x, y, w, h in face_rects:
            cv2.rectangle(draw_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        for obj_id, (c_x, c_y) in objs.items():
            x = c_x - frame_x
            y = c_y - frame_y
            cv2.putText(draw_image, str(obj_id),
                        (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)
            cv2.circle(draw_image, (x, y), 4, (0, 255, 0), -1)
        cv2.imwrite(os.path.join(detections_dir, '{id}.png'.format(id=uuid.uuid4())), draw_image)

    logger.info('init queue consumer')
    with Consumer(connection, queue, callbacks=[handle_message]):
        logger.info('consume queue messages')
        for _ in eventloop(connection):
            pass
