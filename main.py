from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import imutils
import time
import dlib
import cv2
import os
import math
from pygame import mixer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Initialize sound mixer
mixer.init()
sound1 = mixer.Sound('wake_up.mp3')
sound2 = mixer.Sound('alert.mp3')

try:
    sound3 = mixer.Sound('head_pose_alert.mp3')
    use_head_sound = True
except FileNotFoundError:
    print("\u26a0\ufe0f Warning: 'head_pose_alert.mp3' not found. Head pose alerts will be text-only.")
    use_head_sound = False

# Load CNN model from model/ directory
cnn_model = load_model('model/eye_state_cnn_model.h5')

# Alarm flags
drowsy_alarm_on = False
yawn_alarm_on = False
headpose_alarm_on = False

def alarm(msg, alarm_type):
    global drowsy_alarm_on, yawn_alarm_on, headpose_alarm_on

    if alarm_type == "drowsiness" and not drowsy_alarm_on:
        drowsy_alarm_on = True
        print("\u26a0\ufe0f DROWSINESS ALERT!")
        sound1.play()
        time.sleep(2)
        drowsy_alarm_on = False

    elif alarm_type == "yawn" and not yawn_alarm_on:
        yawn_alarm_on = True
        print("\u26a0\ufe0f YAWN ALERT!")
        sound2.play()
        time.sleep(2)
        yawn_alarm_on = False

    elif alarm_type == "head_tilt" and not headpose_alarm_on:
        headpose_alarm_on = True
        print("\u26a0\ufe0f HEAD NOD ALERT!")
        if use_head_sound:
            sound3.play()
        time.sleep(2)
        headpose_alarm_on = False

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    ear = (leftEAR + rightEAR) / 2.0
    return ear, leftEye, rightEye

def lip_distance(shape):
    top_lip = np.concatenate((shape[50:53], shape[61:64]))
    low_lip = np.concatenate((shape[56:59], shape[65:68]))
    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)
    return abs(top_mean[1] - low_mean[1])

def compute_pitch(shape):
    image_points = np.array([
        shape[30], shape[8], shape[36], shape[45], shape[48], shape[54]
    ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0)
    ])

    size = (450, 450)
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4,1))
    _, rotation_vector, _ = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
    rvec_matrix = cv2.Rodrigues(rotation_vector)[0]
    pitch = math.degrees(math.asin(-rvec_matrix[2][1]))
    return pitch

# Constants
YAWN_THRESH = 30
YAWN_CONSEC_FRAMES = 30
HEAD_ALERT_COOLDOWN = 5
YAWN_COUNTER = 0
baseline_pitch_angle = None
last_head_alert_time = 0
EAR_CALIBRATION_FRAMES = 100
pitch_angles = []
ear_calibrated = False

# Load detectors
print("📌 Loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Camera
print("📷 Starting Video Stream...")
vs = VideoStream(src=0).start()
time.sleep(2)

cnn_input_size = (24, 24)
calibration_ears = []
CNN_EYE_THRESH = None
cnn_counter = 0
CNN_EYE_CONSEC_FRAMES = 35

while True:
    frame = vs.read()
    if frame is None:
        continue

    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        ear, leftEye, rightEye = final_ear(shape)
        distance = lip_distance(shape)
        pitch = compute_pitch(shape)

        # Get CNN input from eyes
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        leftEyeHull = cv2.boundingRect(np.array([shape[lStart:lEnd]]))
        rightEyeHull = cv2.boundingRect(np.array([shape[rStart:rEnd]]))

        x, y, w, h = leftEyeHull
        left_eye_img = gray[y:y+h, x:x+w]
        x, y, w, h = rightEyeHull
        right_eye_img = gray[y:y+h, x:x+w]

        try:
            left_eye_resized = cv2.resize(left_eye_img, cnn_input_size)
            right_eye_resized = cv2.resize(right_eye_img, cnn_input_size)
        except:
            continue

        left_input = img_to_array(left_eye_resized.reshape(24,24,1)/255.0)
        right_input = img_to_array(right_eye_resized.reshape(24,24,1)/255.0)

        left_pred = cnn_model.predict(np.expand_dims(left_input, axis=0))[0][0]
        right_pred = cnn_model.predict(np.expand_dims(right_input, axis=0))[0][0]

        eye_state = (left_pred + right_pred) / 2.0

        if not ear_calibrated:
            calibration_ears.append(eye_state)
            pitch_angles.append(pitch)
            if len(calibration_ears) >= EAR_CALIBRATION_FRAMES:
                CNN_EYE_THRESH = np.mean(calibration_ears) * 0.85
                baseline_pitch_angle = np.mean(pitch_angles)
                ear_calibrated = True
                print(f"✅ CNN Eye State Calibrated: Threshold={CNN_EYE_THRESH:.3f}")
                print(f"✅ Pitch Baseline Calibrated: {baseline_pitch_angle:.2f} degrees")
        else:
            if eye_state < CNN_EYE_THRESH:
                cnn_counter += 1
                if cnn_counter >= CNN_EYE_CONSEC_FRAMES:
                    Thread(target=alarm, args=("DROWSINESS", "drowsiness")).start()
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cnn_counter = 0

        if distance > YAWN_THRESH:
            YAWN_COUNTER += 1
            if YAWN_COUNTER >= YAWN_CONSEC_FRAMES:
                Thread(target=alarm, args=("YAWNING", "yawn")).start()
                cv2.putText(frame, "YAWN ALERT!", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            YAWN_COUNTER = 0

        current_time = time.time()
        pitch_diff = baseline_pitch_angle - pitch

        if pitch_diff > 15 and (current_time - last_head_alert_time > HEAD_ALERT_COOLDOWN):
            last_head_alert_time = current_time
            Thread(target=alarm, args=("HEAD NOD", "head_tilt")).start()
            cv2.putText(frame, "HEAD NOD ALERT!", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(frame, f"YAWN: {distance:.2f}", (300, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, f"EAR: {eye_state:.2f}", (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, f"Pitch: {pitch:.2f}", (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("Driver Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
