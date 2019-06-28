import logging
import os
import socket
import uuid
import cv2
from kombu import Consumer

from core import Frame, Timer
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

    face_detector_config = config['face_detector']
    logger.info('init face detector: {face_detector_config}'
                .format(face_detector_config=face_detector_config))
    face_detector = Face(**face_detector_config)

    centroid_tracker_config = config['centroid_tracker']
    logger.info('init centroid tacker: {centroid_tracker_config}'
                .format(centroid_tracker_config=centroid_tracker_config))
    centroid_tracker = CentroidTracker(**centroid_tracker_config)

    def handle_message(body, message):
        with Timer() as tm:
            frame = Frame.from_json(body)
        logger.info('dequeue frame: {frame_size}, {time}'.format(frame_size=frame.size, time=tm))

        with Timer() as tm:
            face_rects = face_detector.detect(frame.image)
        logger.info('face detections count: {face_rects_count}, {time}'
                    .format(face_rects_count=len(face_rects), time=str(tm)))

        with Timer() as tm:
            frame_x, frame_y = frame.position
            face_origin_rects = [(frame_x + x, frame_y + y, w, h) for x, y, w, h in face_rects]
            objs = centroid_tracker.update(face_origin_rects)
        logger.info('update centroid tacker frame: {frame_size}, faces: {face_rects_count}, {time}'
                    .format(frame_size=frame.size, face_rects_count=len(face_rects), time=tm))

        with Timer() as tm:
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
            detection_filename = os.path.join(detections_dir, '{id}.png'.format(id=uuid.uuid4()))
            cv2.imwrite(detection_filename, draw_image)
        logger.info('write detection file: {detection_filename}, {time}'
                    .format(detection_filename=detection_filename, time=tm))

        message.ack()

    logger.info('init queue consumer')
    with Consumer(connection, queue, callbacks=[handle_message]):
        logger.info('consume queue messages')
        while True:
            try:
                connection.drain_events(timeout=1)
            except socket.timeout:
                pass
