import logging.config
import cv2.cv2 as cv2

from detection import Face
from tracking import CentroidTracker


logger = logging.getLogger('watch')


def watch_loop(config, frame_queue, detector_sizes):
    logging.config.dictConfig(config['logging'])

    logger.info('init face detectors: {detector_sizes_count}'.format(detector_sizes_count=len(detector_sizes)))
    face_detectors = [Face(size, **config['face_detector']) for size in detector_sizes]

    logger.info('init centroid tacker')
    centroid_tracker = CentroidTracker(**config['centroid_tracker'])

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

        logger.info('update centroid tacker frame: {frame_counter} {frame_size}, faces: {face_rects_count}'
                    .format(frame_counter=frame_counter, frame_size=frame.size, face_rects_count=len(face_rects)))
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
        cv2.imwrite('samples/images/{frame_counter}.png'.format(frame_counter=frame_counter), draw_image)
