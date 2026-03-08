import time
from collections import deque

import cv2
import mediapipe as mp
from ultralytics import YOLO


mpFace = mp.solutions.face_detection
faceDetector = mpFace.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.60
)

phoneModel = YOLO("yolov8n.pt", task="detect")

recentFaceSeen = deque(maxlen=20)
recentGazeAway = deque(maxlen=24)

phoneVisibleFrames = 0
lastFaceBox = None
lastPhoneCheckFrame = -100
lastPhoneVisible = False
lastFaceSeenFrame = -100

phoneThreshold = 3
gazeMajorityThreshold = 16
faceMissingWindow = 20
faceMissingRequired = 16
faceRecoveryNeeded = 3
gazeDiffThreshold = 34.0
minFaceWidth = 80
minFaceHeight = 80
faceGraceFrames = 10


def detectFace(frame):
    rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = faceDetector.process(rgbFrame)

    if not results.detections:
        return None

    detection = results.detections[0]
    bbox = detection.location_data.relative_bounding_box

    h, w, i = frame.shape

    x = max(0, int(bbox.xmin * w))
    y = max(0, int(bbox.ymin * h))
    bw = max(1, int(bbox.width * w))
    bh = max(1, int(bbox.height * h))

    if bw < minFaceWidth or bh < minFaceHeight:
        return None

    return (x, y, bw, bh)


def detectPhone(frame):
    results = phoneModel(frame, verbose=False)[0]

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if cls == 67 and conf >= 0.30:
            return True

    return False


def detectGaze(frame, faceBox):
    if faceBox is None:
        return False, 0.0

    x, y, w, h = faceBox
    hFrame, wFrame, _ = frame.shape

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(wFrame, x + w)
    y2 = min(hFrame, y + h)

    face = frame[y1:y2, x1:x2]

    if face.size == 0:
        return False, 0.0

    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    if gray.shape[1] < 30:
        return False, 0.0

    left = gray[:, :gray.shape[1] // 2]
    right = gray[:, gray.shape[1] // 2:]

    diff = abs(float(left.mean()) - float(right.mean()))

    return diff > gazeDiffThreshold, diff


def processFrame(frame, frameIndex):
    global phoneVisibleFrames
    global lastFaceBox
    global lastPhoneCheckFrame
    global lastPhoneVisible
    global lastFaceSeenFrame

    faceBox = detectFace(frame)

    if faceBox is not None:
        lastFaceBox = faceBox
        lastFaceSeenFrame = frameIndex
        recentFaceSeen.append(1)
    else:
        if frameIndex - lastFaceSeenFrame <= faceGraceFrames:
            recentFaceSeen.append(1)
        else:
            recentFaceSeen.append(0)

    faceMissingVotes = recentFaceSeen.count(0)

    if len(recentFaceSeen) == faceMissingWindow and faceMissingVotes >= faceMissingRequired:
        recentFaceSeen.clear()
        return {
            "timestamp": int(time.time() * 1000),
            "eventType": "faceMissing",
            "signalScore": faceMissingVotes / float(faceMissingWindow),
            "riskScore": 0.22
        }

    if frameIndex - lastPhoneCheckFrame >= 3:
        lastPhoneVisible = detectPhone(frame)
        lastPhoneCheckFrame = frameIndex

    if lastPhoneVisible:
        phoneVisibleFrames += 1
    else:
        phoneVisibleFrames = max(0, phoneVisibleFrames - 1)

    if phoneVisibleFrames >= phoneThreshold:
        phoneVisibleFrames = 0
        return {
            "timestamp": int(time.time() * 1000),
            "eventType": "phoneDetected",
            "signalScore": 1.0,
            "riskScore": 0.85
        }

    gazeAway, gazeDiff = detectGaze(frame, lastFaceBox)

    if gazeAway:
        recentGazeAway.append(1)
    else:
        recentGazeAway.append(0)

    if len(recentGazeAway) == recentGazeAway.maxlen and sum(recentGazeAway) >= gazeMajorityThreshold:
        recentGazeAway.clear()
        return {
            "timestamp": int(time.time() * 1000),
            "eventType": "gazeAway",
            "signalScore": sum(recentGazeAway) / 24.0 if len(recentGazeAway) > 0 else min(1.0, gazeDiff / 50.0),
            "riskScore": 0.10
        }

    return None