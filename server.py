import logging
import base64
import numpy as np
import cv2
import uvicorn

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from vision import processFrame
from workers import initWorker


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

app = FastAPI(title="ProctorVision")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

frameCounter = 0


@app.on_event("startup")
def startup():

    logging.info("Starting ProctorVision backend")

    worker = initWorker()

    app.state.behaviorWorker = worker

    logging.info("Ray worker initialized")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/sessionSummary")
async def getSessionSummary(request: Request):

    worker = request.app.state.behaviorWorker

    if worker is None:
        return {
            "riskScore": 0,
            "totalEvents": 0,
            "eventTypes": [],
            "events": []
        }

    return await worker.getSessionSummary.remote()


@app.post("/processFrame")
async def processFrameApi(request: Request):

    global frameCounter

    data = await request.json()

    frameBase64 = data.get("frame")

    if frameBase64 is None:
        return {"status": "noFrame"}

    frameBytes = base64.b64decode(frameBase64)

    npFrame = np.frombuffer(frameBytes, np.uint8)

    frame = cv2.imdecode(npFrame, cv2.IMREAD_COLOR)

    if frame is None:
        return {"status": "decodeFailed"}

    event = processFrame(frame, frameCounter)

    frameCounter += 1

    if event is not None:

        worker = request.app.state.behaviorWorker

        logging.info(
            f"EVENT | type={event['eventType']} risk={event['riskScore']}"
        )

        worker.processFeatures.remote(event)

    return {"status": "ok"}


if __name__ == "__main__":

    logging.info("Launching API server")

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )