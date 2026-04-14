import cv2
import mediapipe as mp
import numpy as np
import time
import pygame
import serial
from twilio.rest import Client

# ---------- CONFIG ----------
COM_PORT = 'COM3'   
ACCOUNT_SID = "AC6d6ee22471a6bfdb6e70c4e5eb5fc73d"
AUTH_TOKEN = "1afe65779a4d7e3e48a7620d7c0a06af"
FROM_NUMBER = "+15186174578"
TO_NUMBER = "+919791568567"

# ---------- SERIAL ----------
arduino = serial.Serial('COM3', 9600)
time.sleep(2)

# ---------- TWILIO ----------
client = Client(ACCOUNT_SID, AUTH_TOKEN)

def send_sms():
    try:
        client.messages.create(
            body="🚨 Emergency! Driver inactive on steering.",
            from_=FROM_NUMBER,
            to=TO_NUMBER
        )
        print("SMS Sent")
    except Exception as e:
        print("SMS Error:", e)

# ---------- INIT ----------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera error")
    exit()

pygame.mixer.init()
alarm = pygame.mixer.Sound("alarm.mp3")

# ---------- LANDMARKS ----------
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
NOSE = 1

# ---------- FUNCTIONS ----------
def ear_calc(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

# ---------- SETTINGS ----------
eye_thresh = 0.20
eye_time = 2.5

head_drop_thresh = 0.08
head_time = 3

no_hand_limit = 4
max_events = 4

# ---------- VARIABLES ----------
sleep_start = None
head_start = None
baseline_nose = None

last_key_time = time.time()
no_hand_events = 0
event_active = False
sms_sent = False

alarm_on = False

# ---------- LOOP ----------
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    key = cv2.waitKey(1)

    # SPACE = hand on steering
    if key == 32:
        last_key_time = time.time()
        event_active = False

    no_hand_time = time.time() - last_key_time

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    status = "SAFE"
    color = (0, 255, 0)
    alert_triggered = False

    # ---------- FACE DETECTION ----------
    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:

            h, w, _ = frame.shape

            # EYE DETECTION
            left_eye = np.array([(int(face_landmarks.landmark[i].x*w),
                                  int(face_landmarks.landmark[i].y*h)) for i in LEFT_EYE])
            right_eye = np.array([(int(face_landmarks.landmark[i].x*w),
                                   int(face_landmarks.landmark[i].y*h)) for i in RIGHT_EYE])

            ear = (ear_calc(left_eye) + ear_calc(right_eye)) / 2.0

            if ear < eye_thresh:
                if sleep_start is None:
                    sleep_start = time.time()
                elif time.time() - sleep_start > eye_time:
                    status = "EYE ALERT"
                    color = (0, 0, 255)
                    alert_triggered = True
            else:
                sleep_start = None

            # HEAD DETECTION
            nose = face_landmarks.landmark[NOSE]
            ny = nose.y
            nx = nose.x

            if baseline_nose is None:
                baseline_nose = ny

            if 0.3 < nx < 0.7:
                drop = ny - baseline_nose

                if drop > head_drop_thresh:
                    if head_start is None:
                        head_start = time.time()
                    elif time.time() - head_start > head_time:
                        status = "HEAD DOWN"
                        color = (0, 0, 255)
                        alert_triggered = True
                else:
                    head_start = None
            else:
                head_start = None

    else:
        status = "NO FACE"
        color = (0, 0, 255)
        alert_triggered = True

    # ---------- STEERING ----------
    if no_hand_time > no_hand_limit:
        status = "NO HAND"
        color = (0, 0, 255)
        alert_triggered = True

        if not event_active:
            no_hand_events += 1
            print("Event:", no_hand_events)
            event_active = True

        if no_hand_events >= max_events and not sms_sent:
            send_sms()
            sms_sent = True

    # ---------- ALERT CONTROL ----------
    if alert_triggered and not alarm_on:
        pygame.mixer.Sound.play(alarm, loops=-1)
        alarm_on = True
        arduino.write(b'1')

    elif not alert_triggered and alarm_on:
        alarm.stop()
        alarm_on = False
        arduino.write(b'0')

    # ---------- DISPLAY ----------
    cv2.putText(frame, status, (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

    cv2.putText(frame, f"No Hand: {round(no_hand_time,1)}s", (50, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

    cv2.putText(frame, f"Events: {no_hand_events}", (50, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.imshow("Driver Monitor", frame)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()