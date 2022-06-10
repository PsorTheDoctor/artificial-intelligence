import cv2
import mediapipe as mp
import time

mp_objectron = mp.solutions.objectron
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Available models are: Shoe, Chair, Cup, Camera
with mp_objectron.Objectron(static_image_mode=False,
                            max_num_objects=2,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.8,
                            model_name='Shoe') as objectron:
    while cap.isOpened():
        (_, frame) = cap.read()
        start = time.time()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False
        results = objectron.process(frame)

        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if results.detected_objects:
            for obj in results.detected_objects:
                mp_drawing.draw_landmarks(frame, obj.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
                mp_drawing.draw_axis(frame, obj.rotation, obj.translation)

        end = time.time()
        fps = 1 / (end - start)

        cv2.putText(frame, f'FPS: {int(fps)}', (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Objectron', frame)

        if cv2.waitKey(50) == 27:
            break

cap.release()
cv2.destroyAllWindows()
