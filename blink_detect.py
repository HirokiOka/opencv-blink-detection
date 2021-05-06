import os,sys
from scipy.spatial import distance
import cv2
import dlib
from imutils import face_utils

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('./haarcascaede_frontalface_alt2.xml')
face_parts_detector = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
cvFont = cv2.FONT_HERSHEY_PLAIN

def calc_ear(eye_c):
    a = distance.euclidean(eye_c[1], eye_c[5])
    b = distance.euclidean(eye_c[2], eye_c[4])
    c = distance.euclidean(eye_c[0], eye_c[3])
    ear = (a + b) / (c * 2.0)
    return round(ear, 3)

def detect_eye_close(right_ear, left_ear):
    if ((right_ear < 0.25) and (left_ear <0.25)):
        return True
    else:
        return False

def show_params(eye_state, tick, rgb, blink_count):
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - tick)
    cv2.putText(rgb, "FPS:{} ".format(int(fps)), (10, 50),
        cvFont, 3, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(rgb, "Blink Count:{} ".format(blink_count), (10, 80),
            cvFont, 2, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(rgb, "Eye:{} ".format(eye_state), (10, 120),
            cvFont, 2, (0, 0, 0), 2, cv2.LINE_AA)

def main():
    blink_count = 0
    eye_state = "OPEN"
    pre_is_eye_closed = False

    while True:
        tick = cv2.getTickCount()

        ret, rgb = cap.read()
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.11, minNeighbors=3, minSize=(100, 100))

        if len(faces) == 1:
            x, y, w, h = faces[0, :]
            cv2.rectangle(rgb, (x, y), (x + w, y + h), (0, 0, 255), 2)

            face = dlib.rectangle(x, y, x + w, y + h)
            face_parts = face_parts_detector(gray, face)
            face_parts = face_utils.shape_to_np(face_parts)

            right_ear = calc_ear(face_parts[42:48])
            left_ear = calc_ear(face_parts[36:42])

            is_eye_closed = detect_eye_close(right_ear, left_ear)

            if (is_eye_closed and (is_eye_closed != pre_is_eye_closed)):
                blink_count+=1

            pre_is_eye_closed = is_eye_closed

            for i, ((x, y)) in enumerate(face_parts[36:48]):
                cv2.circle(rgb, (x, y), 1, (0, 255, 0), -1)
                cv2.putText(rgb, str(i), (x + 2, y - 2),
                        cvFont, 0.3, (0, 255, 0), 1)

        else:
            cv2.putText(rgb, "Face Detection Failed", (10, 120),
                    cvFont, 2, (0, 0, 255), 2, cv2.LINE_AA)

        if (is_eye_closed):
            eye_state = "CLOSE"
        else:
            eye_state = "OPEN"

        show_params(eye_state, tick, rgb, blink_count)
        cv2.imshow('frame', rgb)

        if cv2.waitKey(1) == 27:
            break


if __name__ == '__main__':
    main()
    cap.release()
    cv2.destroyAllWindows()
