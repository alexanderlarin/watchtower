import logging.config
import cv2.cv2 as cv2

from detection import Face
from tracking import CentroidTracker


logger = logging.getLogger('watch')


def watch_loop(config, frame_queue, detector_sizes):
    logging.config.dictConfig(config['logging'])

    logger.info(f'init face detectors: {len(detector_sizes)}')
    face_detectors = [Face(size,
                           'network_models/deploy.prototxt',
                           'network_models/res10_300x300_ssd_iter_140000.caffemodel',
                           (104.0, 177.0, 123.0),
                           threshold=.4, scale_factor=1.) for size in detector_sizes]

    logger.info('init face tacker')
    face_tracker = CentroidTracker(max_disappeared=50, max_distance=75)

    frame_counter = 0
    while True:
        if frame_queue.empty():
            continue
        frame = frame_queue.get()
        if frame is None:
            break
        frame_counter = frame_counter + 1
        face_detector = next((d for d in face_detectors if d.image_size == frame.size), None)
        face_rects = face_detector.detect(frame.image)
        frame_x, frame_y = frame.position
        face_origin_rects = [(frame_x + x, frame_y + y, w, h) for x, y, w, h in face_rects]

        logger.info(f'update face tacker frame: {frame_counter} {frame.size}, faces: {len(face_rects)}')
        objs = face_tracker.update(face_origin_rects)

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
        cv2.imwrite(f'samples/images/{frame_counter}.png', draw_image)
