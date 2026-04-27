import time
import logging
from collections import deque
from dataclasses import dataclass

import ray

logger = logging.getLogger(__name__)


@dataclass
class RayActorConfig:
    behaviorWorkerCpu: float = 1
    behaviorWorkerGpu: float = 0
    behaviorWorkerMemory: float = 500e6
    maxPendingTasks: int = 300


def initRayCluster(localMode: bool = True, numWorkers: int = 1, useGpu: bool = False):
    if ray.is_initialized():
        return getClusterInfo()

    ray.init(
        ignore_reinit_error=True,
        num_cpus=max(numWorkers, 1),
        num_gpus=1 if useGpu else 0,
        include_dashboard=False,
        log_to_driver=False,
    )

    info = getClusterInfo()
    logger.info(f"Ray initialized: {info}")
    return info


def init_ray_cluster(local_mode: bool = True, num_workers: int = 1, use_gpu: bool = False):
    return initRayCluster(localMode=local_mode, numWorkers=num_workers, useGpu=use_gpu)


def getClusterInfo():
    if not ray.is_initialized():
        return None

    resources = ray.cluster_resources()
    return {
        "cpus": resources.get("CPU", 0),
        "gpus": resources.get("GPU", 0),
        "memoryGb": round(resources.get("memory", 0) / 1e9, 2),
    }


def shutdownRay():
    if ray.is_initialized():
        ray.shutdown()
        logger.info("Ray shut down")


def shutdown_ray():
    shutdownRay()


def getActorOptions():
    return {
        "num_cpus": RayActorConfig.behaviorWorkerCpu,
        "num_gpus": RayActorConfig.behaviorWorkerGpu,
        "memory": RayActorConfig.behaviorWorkerMemory,
    }


@ray.remote(**getActorOptions())
class BehaviorWorker:
    def __init__(self):
        self.events = []
        self.riskScore = 0.0
        self.sessionStart = time.time()
        self.frameCount = 0
        self.processingTimesMs = []

        self.recentGazeAway = deque(maxlen=24)
        self.recentFaceSeen = deque(maxlen=20)
        self.recentPhoneDetected = deque(maxlen=6)
        self.recentMultipleFaces = deque(maxlen=8)

        self.lastGazeAwayEvent = 0
        self.lastFaceMissingEvent = 0
        self.lastPhoneEvent = 0
        self.lastMultipleFacesEvent = 0

        self.gazeAwayCooldownSec = 20
        self.faceMissingCooldownSec = 25
        self.phoneCooldownSec = 8
        self.multipleFacesCooldownSec = 20

    def processFrame(self, detections: dict) -> dict:
        startTime = time.time()
        currentTime = time.time()
        self.frameCount += 1

        frameNum = detections.get("frameNum", self.frameCount)
        faceCount = detections.get("faceCount", 0)
        gazeAway = detections.get("gazeAway", detections.get("gazeAwayAngle", 0.0) > 35.0)
        phoneDetected = bool(detections.get("phoneDetected", False))

        self.recentGazeAway.append(bool(gazeAway))
        self.recentFaceSeen.append(faceCount > 0)
        self.recentPhoneDetected.append(phoneDetected)
        self.recentMultipleFaces.append(faceCount > 1)

        firedEvents = []
        riskDelta = 0.0

        if len(self.recentGazeAway) >= 16 and sum(self.recentGazeAway) >= 16:
            if currentTime - self.lastGazeAwayEvent > self.gazeAwayCooldownSec:
                firedEvents.append(("gazeAway", 0.10))
                self.lastGazeAwayEvent = currentTime

        faceMissingCount = sum(1 for seen in self.recentFaceSeen if not seen)

        if len(self.recentFaceSeen) >= 16 and faceMissingCount >= 16:
            if currentTime - self.lastFaceMissingEvent > self.faceMissingCooldownSec:
                firedEvents.append(("faceMissing", 0.15))
                self.lastFaceMissingEvent = currentTime

        if len(self.recentPhoneDetected) >= 3 and sum(self.recentPhoneDetected) >= 2:
            if currentTime - self.lastPhoneEvent > self.phoneCooldownSec:
                firedEvents.append(("phoneDetected", 0.40))
                self.lastPhoneEvent = currentTime

        if len(self.recentMultipleFaces) >= 4 and sum(self.recentMultipleFaces) >= 4:
            if currentTime - self.lastMultipleFacesEvent > self.multipleFacesCooldownSec:
                firedEvents.append(("multipleFaces", 0.35))
                self.lastMultipleFacesEvent = currentTime

        for eventType, delta in firedEvents:
            riskDelta += delta
            self.riskScore = min(1.0, self.riskScore + delta)
            self.events.append({
                "timestamp": int(time.time() * 1000),
                "frameNum": frameNum,
                "eventType": eventType,
                "riskDelta": delta,
                "riskScore": round(self.riskScore, 3),
            })

        processingTimeMs = (time.time() - startTime) * 1000
        self.processingTimesMs.append(processingTimeMs)

        if len(self.processingTimesMs) > 100:
            self.processingTimesMs.pop(0)

        return {
            "frameNum": frameNum,
            "eventFired": len(firedEvents) > 0,
            "eventType": firedEvents[-1][0] if firedEvents else None,
            "events": [eventType for eventType, _ in firedEvents],
            "riskDelta": round(riskDelta, 3),
            "riskScore": round(self.riskScore, 3),
            "processingTimeMs": round(processingTimeMs, 2),
        }

    def getSessionSummary(self) -> dict:
        recentEvents = self.events[-20:]
        eventTypes = list(dict.fromkeys(event["eventType"] for event in recentEvents))

        avgProcessingTimeMs = (
            sum(self.processingTimesMs) / len(self.processingTimesMs)
            if self.processingTimesMs else 0.0
        )

        return {
            "riskScore": round(self.riskScore, 3),
            "totalEvents": len(self.events),
            "eventTypes": eventTypes,
            "events": recentEvents,
            "framesProcessed": self.frameCount,
            "elapsedSec": round(time.time() - self.sessionStart, 1),
            "avgProcessingTimeMs": round(avgProcessingTimeMs, 2),
            "maxProcessingTimeMs": round(max(self.processingTimesMs) if self.processingTimesMs else 0.0, 2),
        }

    def reset(self):
        self.events.clear()
        self.riskScore = 0.0
        self.sessionStart = time.time()
        self.frameCount = 0
        self.processingTimesMs.clear()
        self.recentGazeAway.clear()
        self.recentFaceSeen.clear()
        self.recentPhoneDetected.clear()
        self.recentMultipleFaces.clear()
        return {"status": "reset"}


class BehaviorWorkerPool:
    def __init__(self, numWorkers: int = 1):
        self.numWorkers = max(1, numWorkers)
        self.workers = [BehaviorWorker.remote() for _ in range(self.numWorkers)]
        self.pendingTasks = {}
        self.nextWorkerIndex = 0
        logger.info(f"Spawned {self.numWorkers} BehaviorWorker actors")

    def submit_frame(self, detections: dict):
        return self.submitFrame(detections)

    def submitFrame(self, detections: dict):
        frameNum = detections.get("frameNum", int(time.time() * 1000))

        if len(self.pendingTasks) > RayActorConfig.maxPendingTasks:
            oldestFrameNum = min(self.pendingTasks.keys())
            del self.pendingTasks[oldestFrameNum]

        worker = self.workers[self.nextWorkerIndex]
        self.nextWorkerIndex = (self.nextWorkerIndex + 1) % self.numWorkers

        taskRef = worker.processFrame.remote(detections)
        self.pendingTasks[frameNum] = taskRef

        return frameNum

    def get_result(self, frameNum: int, timeout_sec: float = 0.0):
        return self.getResult(frameNum, timeoutSec=timeout_sec)

    def getResult(self, frameNum: int, timeoutSec: float = 0.0):
        if frameNum not in self.pendingTasks:
            return {
                "status": "missing",
                "frameNum": frameNum,
                "ready": False,
            }

        taskRef = self.pendingTasks[frameNum]
        readyRefs, _ = ray.wait([taskRef], timeout=timeoutSec)

        if not readyRefs:
            return {
                "status": "processing",
                "frameNum": frameNum,
                "ready": False,
            }

        result = ray.get(taskRef)
        del self.pendingTasks[frameNum]
        result["status"] = "ok"
        result["ready"] = True
        return result

    def get_all_summaries(self):
        return self.getAllSummaries()

    def getAllSummaries(self):
        summaryRefs = [worker.getSessionSummary.remote() for worker in self.workers]
        summaries = ray.get(summaryRefs)

        events = []
        eventTypes = set()

        for summary in summaries:
            events.extend(summary.get("events", []))
            eventTypes.update(summary.get("eventTypes", []))

        events.sort(key=lambda event: event.get("timestamp", 0))

        return {
            "riskScore": round(max((summary.get("riskScore", 0) for summary in summaries), default=0), 3),
            "totalEvents": sum(summary.get("totalEvents", 0) for summary in summaries),
            "framesProcessed": sum(summary.get("framesProcessed", 0) for summary in summaries),
            "eventTypes": list(eventTypes),
            "events": events[-20:],
            "pendingTasks": len(self.pendingTasks),
            "workers": summaries,
        }

    def reset_all(self):
        return self.resetAll()

    def resetAll(self):
        resetRefs = [worker.reset.remote() for worker in self.workers]
        ray.get(resetRefs)
        self.pendingTasks.clear()
        return {"status": "reset"}

    def shutdown(self):
        self.workers.clear()
        self.pendingTasks.clear()
        logger.info("Worker pool shut down")
