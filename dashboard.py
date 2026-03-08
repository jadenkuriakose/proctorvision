import streamlit as st
import requests
import time
from datetime import datetime
import streamlit.components.v1 as components

st.set_page_config(
    page_title="ProctorVision",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

apiUrl = "http://localhost:8000"


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    background-color:#1a1a1a;
    color:#ececec;
    font-family:'Inter', sans-serif;
}

.metricCard{
    background:#242424;
    border:1px solid #333;
    border-radius:10px;
    padding:1.4rem 1.6rem;
    margin-bottom:1rem;
}

.metricLabel{
    font-size:0.72rem;
    color:#888;
    text-transform:uppercase;
    letter-spacing:0.08em;
}

.metricValue{
    font-size:2rem;
    font-weight:500;
}

.sectionHeader{
    font-size:0.72rem;
    color:#666;
    text-transform:uppercase;
    letter-spacing:0.1em;
    margin-bottom:1rem;
}

.eventRow{
    display:flex;
    justify-content:space-between;
    border-bottom:1px solid #2a2a2a;
    padding:0.6rem 0;
}

.eventBadge{
    padding:3px 10px;
    border-radius:20px;
    font-size:0.72rem;
}

.badgeFaceMissing{background:#2a1010;color:#f87171}
.badgePhoneDetected{background:#1e1030;color:#a78bfa}
.badgeGazeAway{background:#2a2010;color:#e8b84b}

#MainMenu, footer, header { visibility:hidden; }

.statusDot{
    width:7px;
    height:7px;
    border-radius:50%;
    display:inline-block;
    margin-right:6px;
    background:#34d399;
}
</style>
""", unsafe_allow_html=True)


badgeClasses = {
    "gazeAway":"badgeGazeAway",
    "phoneDetected":"badgePhoneDetected",
    "faceMissing":"badgeFaceMissing"
}

badgeLabels = {
    "gazeAway":"Gaze Away",
    "phoneDetected":"Phone Detected",
    "faceMissing":"Face Missing"
}


def fetchSummary():

    try:
        r = requests.get(f"{apiUrl}/sessionSummary",timeout=1)
        return r.json()

    except:
        return {
            "riskScore":0,
            "totalEvents":0,
            "eventTypes":[],
            "events":[]
        }


def renderBadge(eventType):

    css = badgeClasses.get(eventType,"badgeGazeAway")
    label = badgeLabels.get(eventType,eventType)

    return f'<span class="eventBadge {css}">{label}</span>'


def formatTimestamp(ts):

    try:
        return datetime.fromtimestamp(ts/1000).strftime("%H:%M:%S")
    except:
        return "—"


st.markdown("""
<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:2rem;">
<div>
<div style="font-size:1.5rem;font-weight:500;">ProctorVision</div>
<div style="font-size:0.8rem;color:#666;">Real-time exam monitoring</div>
</div>
<div style="font-size:0.8rem;color:#888;">
<span class="statusDot"></span>Live Session
</div>
</div>
""", unsafe_allow_html=True)


col1,col2 = st.columns([2,1],gap="large")


# ---------- CAMERA ----------
with col1:

    st.markdown('<div class="sectionHeader">Live Feed</div>', unsafe_allow_html=True)

    components.html(
        f"""
        <video id="video" autoplay playsinline
        style="width:100%;border-radius:10px;border:1px solid #333;"></video>

        <script>

        const video = document.getElementById("video");

        async function startCamera() {{

            const stream = await navigator.mediaDevices.getUserMedia({{video:true}});
            video.srcObject = stream;

            const canvas = document.createElement("canvas");
            const ctx = canvas.getContext("2d");

            async function sendFrame() {{

                if(video.videoWidth === 0) return;

                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;

                ctx.drawImage(video,0,0);

                const data = canvas.toDataURL("image/jpeg",0.7);

                fetch("{apiUrl}/processFrame",{{
                    method:"POST",
                    headers:{{"Content-Type":"application/json"}},
                    body:JSON.stringify({{frame:data.split(",")[1]}})
                }});

            }}

            setInterval(sendFrame,150);
        }}

        startCamera();

        </script>
        """,
        height=520,
    )


# ---------- METRICS ----------
with col2:

    metricsSlot = st.empty()

feedSlot = st.empty()


# ---------- LIVE DASHBOARD LOOP ----------
while True:

    summary = fetchSummary()

    riskScore = summary.get("riskScore",0)
    totalEvents = summary.get("totalEvents",0)
    eventTypes = summary.get("eventTypes",[])
    events = summary.get("events",[])

    riskPct = int(riskScore*100)

    with metricsSlot.container():

        st.markdown('<div class="sectionHeader">Session Metrics</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metricCard">
        <div class="metricLabel">Risk Score</div>
        <div class="metricValue">{riskPct}%</div>
        </div>

        <div class="metricCard">
        <div class="metricLabel">Total Events</div>
        <div class="metricValue">{totalEvents}</div>
        </div>

        <div class="metricCard">
        <div class="metricLabel">Active Signals</div>
        {''.join(renderBadge(e) for e in eventTypes) if eventTypes else '<span style="color:#555">None</span>'}
        </div>
        """, unsafe_allow_html=True)


    with feedSlot.container():

        st.markdown('<div class="sectionHeader">Event Feed</div>', unsafe_allow_html=True)

        if not events:

            st.markdown('<div style="color:#555">No events yet</div>', unsafe_allow_html=True)

        else:

            rows=""

            for event in reversed(events[-15:]):

                badge = renderBadge(event.get("eventType",""))
                ts = formatTimestamp(event.get("timestamp",0))

                rows+=f"""
                <div class="eventRow">
                <div>{badge}</div>
                <div style="color:#777;font-size:0.75rem">{ts}</div>
                </div>
                """

            st.markdown(rows,unsafe_allow_html=True)

    time.sleep(0.3)