import cv2
import mediapipe as mp
from imutils.video import VideoStream

face_detector = mp.solutions.face_detection
draw = mp.solutions.drawing_utils

# Model files can be downloaded here: https://github.com/isl-org/MiDaS/releases/tag/v2_1
model_name = 'model-small.onnx'
model = cv2.dnn.readNet(model_name)


def depth_to_distance(depth):
    return -1.7 * depth + 2


vs = VideoStream(0).start()

with face_detector.FaceDetection(min_detection_confidence=0.6) as face_detection:
    while True:
        frame = vs.read()
        h, w, channels = frame.shape
        results = face_detection.process(frame)

        centroid = None
        if results.detections:
            for id, detection in enumerate(results.detections):
                draw.draw_detection(frame, detection)
                bbox = detection.location_data.relative_bounding_box
                h, w, channels = frame.shape

                boundBox = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)
                centroid = (boundBox[0] + boundBox[2] / 2, boundBox[1] + boundBox[3] / 2)

                cv2.putText(frame, f'{int(detection.score[0] * 100)}%', (boundBox[0], boundBox[1] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Depth map from neural net
        if centroid is not None:
            mean_subtraction = (123.675, 116.28, 103.53)
            blob = cv2.dnn.blobFromImage(frame, 1/255., (256, 256), mean_subtraction, True, False)
            model.setInput(blob)

            depth_map = model.forward()
            depth_map = depth_map[0, :, :]
            depth_map = cv2.resize(depth_map, (w, h))

            depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

            depth_face = depth_map[int(centroid[1]), int(centroid[0])]

            cv2.putText(frame, 'Depth in cm: ' + str(round(depth_face * 100, 2)), (50, 400),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Depth map', depth_map)

        cv2.imshow('Face detection', frame)

        if cv2.waitKey(50) == 27:
            break

cv2.destroyAllWindows()
vs.stop()
