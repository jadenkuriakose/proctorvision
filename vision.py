import time
import os
from collections import deque

import cv2
import mediapipe as mp

try:
    import torch
except Exception:
    torch = None

from ultralytics import YOLO


mpFace = mp.solutions.face_detection
faceDetector = mpFace.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.60
)

modelPath = os.getenv("PHONE_MODEL_PATH", "yolov8n.pt")
phoneImgSize = int(os.getenv("PHONE_IMG_SIZE", "320"))
phoneConfThreshold = float(os.getenv("PHONE_CONF_THRESHOLD", "0.30"))
phoneCheckInterval = int(os.getenv("PHONE_CHECK_INTERVAL", "8"))
phoneHoldFrames = int(os.getenv("PHONE_HOLD_FRAMES", "12"))
faceCheckInterval = int(os.getenv("FACE_CHECK_INTERVAL", "3"))
faceHoldFrames = int(os.getenv("FACE_HOLD_FRAMES", "8"))
useGpu = os.getenv("USE_GPU", "auto").lower()
phoneClassIds = [int(value.strip()) for value in os.getenv("PHONE_CLASS_IDS", "67").split(",") if value.strip()]

gazeDiffThreshold = float(os.getenv("GAZE_DIFF_THRESHOLD", "34.0"))
minFaceWidth = int(os.getenv("MIN_FACE_WIDTH", "70"))
minFaceHeight = int(os.getenv("MIN_FACE_HEIGHT", "70"))

useOnnx = modelPath.endswith(".onnx")
recentFaceSeen = deque(maxlen=20)
recentGazeAway = deque(maxlen=24)

lastFaceBox = None
lastFaceSeenFrame = -100
lastFaceCheckFrame = -100

lastPhoneVisible = False
lastPhoneSeenFrame = -100
lastPhoneCheckFrame = -100
lastPhoneBoxes = []


def getModelDevice():
    if useOnnx:
        return None

    if torch is None:
        return "cpu"

    if useGpu in ["0", "false", "no", "cpu"]:
        return "cpu"

    if torch.cuda.is_available():
        return 0

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


modelDevice = getModelDevice()
phoneModel = YOLO(modelPath, task="detect")

if not useOnnx:
    phoneModel.to(modelDevice)

    try:
        phoneModel.fuse()
    except Exception:
        pass


def detectFace(frame):
    rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = faceDetector.process(rgbFrame)

    if not results.detections:
        return None

    detection = results.detections[0]
    bbox = detection.location_data.relative_bounding_box
    frameHeight, frameWidth, _ = frame.shape

    x = max(0, int(bbox.xmin * frameWidth))
    y = max(0, int(bbox.ymin * frameHeight))
    boxWidth = max(1, int(bbox.width * frameWidth))
    boxHeight = max(1, int(bbox.height * frameHeight))

    if boxWidth < minFaceWidth or boxHeight < minFaceHeight:
        return None

    return (x, y, boxWidth, boxHeight)


def getFaceBox(frame, frameIndex):
    global lastFaceBox
    global lastFaceSeenFrame
    global lastFaceCheckFrame

    faceWasChecked = False

    if frameIndex - lastFaceCheckFrame >= faceCheckInterval:
        faceWasChecked = True
        faceBox = detectFace(frame)
        lastFaceCheckFrame = frameIndex

        if faceBox is not None:
            lastFaceBox = faceBox
            lastFaceSeenFrame = frameIndex
            recentFaceSeen.append(1)
            return faceBox, 1, faceWasChecked

        if frameIndex - lastFaceSeenFrame <= faceHoldFrames:
            recentFaceSeen.append(1)
            return lastFaceBox, 1, faceWasChecked

        recentFaceSeen.append(0)
        return None, 0, faceWasChecked

    if frameIndex - lastFaceSeenFrame <= faceHoldFrames and lastFaceBox is not None:
        recentFaceSeen.append(1)
        return lastFaceBox, 1, faceWasChecked

    recentFaceSeen.append(0)
    return None, 0, faceWasChecked


def detectPhone(frame):
    global lastPhoneBoxes

    smallFrame = cv2.resize(frame, (phoneImgSize, phoneImgSize))

    kwargs = {
        "verbose": False,
        "imgsz": phoneImgSize,
        "conf": phoneConfThreshold,
    }

    if not useOnnx:
        kwargs["device"] = modelDevice
        kwargs["half"] = False

    results = phoneModel(smallFrame, **kwargs)[0]

    boxes = []

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if cls in phoneClassIds and conf >= phoneConfThreshold:
            xyxy = box.xyxy[0].detach().cpu().numpy().tolist()
            boxes.append({
                "classId": cls,
                "conf": round(conf, 3),
                "xyxy": [round(float(value), 2) for value in xyxy],
            })

    lastPhoneBoxes = boxes
    return len(boxes) > 0


def getPhoneSignal(frame, frameIndex):
    global lastPhoneVisible
    global lastPhoneSeenFrame
    global lastPhoneCheckFrame
    global lastPhoneBoxes

    phoneWasChecked = False

    if frameIndex - lastPhoneCheckFrame >= phoneCheckInterval:
        phoneWasChecked = True
        phoneVisible = detectPhone(frame)
        lastPhoneCheckFrame = frameIndex

        if phoneVisible:
            lastPhoneVisible = True
            lastPhoneSeenFrame = frameIndex
            return True, phoneWasChecked

        lastPhoneVisible = False

    if frameIndex - lastPhoneSeenFrame <= phoneHoldFrames:
        return True, phoneWasChecked

    lastPhoneBoxes = []
    return False, phoneWasChecked


def detectGaze(frame, faceBox):
    if faceBox is None:
        return False, 0.0

    x, y, width, height = faceBox
    frameHeight, frameWidth, _ = frame.shape

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(frameWidth, x + width)
    y2 = min(frameHeight, y + height)

    face = frame[y1:y2, x1:x2]

    if face.size == 0:
        return False, 0.0

    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    if gray.shape[1] < 30:
        return False, 0.0

    leftHalf = gray[:, :gray.shape[1] // 2]
    rightHalf = gray[:, gray.shape[1] // 2:]

    diff = abs(float(leftHalf.mean()) - float(rightHalf.mean()))
    return diff > gazeDiffThreshold, diff


def processFrame(frame, frameIndex):
    startTime = time.time()

    faceBox, faceCount, faceWasChecked = getFaceBox(frame, frameIndex)
    gazeAway, gazeDiff = detectGaze(frame, faceBox)
    phoneDetected, phoneWasChecked = getPhoneSignal(frame, frameIndex)

    recentGazeAway.append(1 if gazeAway else 0)

    return {
        "frameNum": frameIndex,
        "timestamp": int(time.time() * 1000),
        "faceCount": faceCount,
        "faceWasChecked": faceWasChecked,
        "gazeAway": gazeAway,
        "gazeAwayAngle": gazeDiff,
        "phoneDetected": phoneDetected,
        "phoneWasChecked": phoneWasChecked,
        "phoneBoxes": lastPhoneBoxes,
        "modelDevice": str(modelDevice),
        "modelPath": modelPath,
        "phoneImgSize": phoneImgSize,
        "visionTimeMs": round((time.time() - startTime) * 1000, 2),
    }