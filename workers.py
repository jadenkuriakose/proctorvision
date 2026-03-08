import time
import ray


@ray.remote
class BehaviorWorker:

    def __init__(self):

        self.events = []
        self.riskScore = 0.0
        self.sessionStart = time.time()

    def processFeatures(self, event):

        eventType = event.get("eventType", "unknown")

        timestamp = event.get("timestamp", int(time.time() * 1000))

        riskDelta = float(event.get("riskScore", 0.1))

        self.riskScore = min(1.0, self.riskScore + riskDelta)

        self.events.append({
            "timestamp": timestamp,
            "eventType": eventType,
            "riskScore": round(self.riskScore, 3)
        })

    def getSessionSummary(self):

        recentEvents = self.events[-20:]

        activeTypes = list(dict.fromkeys(
            [e["eventType"] for e in recentEvents]
        ))

        return {
            "riskScore": round(self.riskScore, 3),
            "totalEvents": len(self.events),
            "eventTypes": activeTypes,
            "events": recentEvents,
            "elapsedSec": round(time.time() - self.sessionStart, 1)
        }


def initWorker():

    if not ray.is_initialized():

        ray.init(
            ignore_reinit_error=True,
            num_cpus=2,
            include_dashboard=False
        )

    return BehaviorWorker.remote()