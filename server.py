import logging
import base64
import os
import warnings
import numpy as np
import cv2
import uvicorn
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from vision import processFrame, modelDevice, modelPath
from workers import BehaviorWorkerPool, init_ray_cluster, shutdown_ray


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

frameCounter = 0
workerPool = None
numWorkers = int(os.getenv("behaviorWorkers", os.getenv("BEHAVIOR_WORKERS", "1")))


@asynccontextmanager
async def lifespan(app: FastAPI):
    global workerPool

    logger.info("Starting ProctorVision backend")

    try:
        init_ray_cluster(local_mode=True, num_workers=numWorkers, use_gpu=False)
        workerPool = BehaviorWorkerPool(numWorkers=numWorkers)
        app.state.workerPool = workerPool
        logger.info("Worker pool ready")
    except Exception as error:
        logger.error(f"Startup failed: {error}")
        raise

    yield

    logger.info("Shutting down ProctorVision backend")

    if workerPool is not None:
        workerPool.shutdown()

    shutdown_ray()


app = FastAPI(
    title="ProctorVision",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "backend": "ray",
        "workers": numWorkers,
        "modelDevice": str(modelDevice),
        "modelPath": modelPath,
    }


@app.get("/sessionSummary")
async def getSessionSummary(request: Request):
    global workerPool

    if workerPool is None:
        return {
            "riskScore": 0,
            "totalEvents": 0,
            "eventTypes": [],
            "events": [],
            "framesProcessed": 0,
            "pendingTasks": 0,
        }

    try:
        return workerPool.get_all_summaries()
    except Exception as error:
        logger.error(f"Session summary error: {error}")
        return {
            "status": "error",
            "message": str(error),
        }


@app.post("/processFrame")
async def processFrameApi(request: Request):
    global frameCounter
    global workerPool

    try:
        data = await request.json()
        frameBase64 = data.get("frame")

        if frameBase64 is None:
            return {
                "status": "noFrame",
                "frameNum": frameCounter,
            }

        if "," in frameBase64:
            frameBase64 = frameBase64.split(",", 1)[1]

        frameBytes = base64.b64decode(frameBase64)
        npFrame = np.frombuffer(frameBytes, np.uint8)
        frame = cv2.imdecode(npFrame, cv2.IMREAD_COLOR)

        if frame is None:
            return {
                "status": "decodeFailed",
                "frameNum": frameCounter,
            }

        currentFrame = frameCounter
        detections = processFrame(frame, currentFrame)

        if not isinstance(detections, dict):
            return {
                "status": "error",
                "message": "vision pipeline failed",
                "frameNum": currentFrame,
            }

        detections["frameNum"] = currentFrame

        if workerPool is None:
            frameCounter += 1
            return {
                "status": "ok",
                "frameNum": currentFrame,
                "detections": detections,
                "eventFired": False,
                "eventType": None,
                "riskScore": 0,
            }

        taskId = workerPool.submit_frame(detections)
        result = workerPool.get_result(taskId, timeout_sec=0.0)

        frameCounter += 1

        response = {
            "status": "processing" if result.get("ready") is False else "ok",
            "frameNum": currentFrame,
            "taskId": taskId,
            "detections": detections,
        }

        if result.get("ready") is not False:
            response.update({
                "eventFired": result.get("eventFired", False),
                "eventType": result.get("eventType"),
                "events": result.get("events", []),
                "riskDelta": result.get("riskDelta", 0),
                "riskScore": result.get("riskScore", 0),
                "processingTimeMs": result.get("processingTimeMs", 0),
            })

        return response

    except Exception as error:
        logger.error(f"Process frame error: {error}")
        return {
            "status": "error",
            "message": str(error),
        }


@app.get("/taskResult/{frameNum}")
async def getTaskResult(frameNum: int, request: Request):
    global workerPool

    if workerPool is None:
        return {
            "status": "error",
            "message": "worker pool unavailable",
            "frameNum": frameNum,
        }

    try:
        return workerPool.get_result(frameNum, timeout_sec=0.0)
    except Exception as error:
        return {
            "status": "error",
            "message": str(error),
            "frameNum": frameNum,
        }


@app.post("/resetSession")
async def resetSession():
    global frameCounter
    global workerPool

    frameCounter = 0

    if workerPool is not None:
        return workerPool.reset_all()

    return {
        "status": "reset",
    }


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
        workers=1,
    )
