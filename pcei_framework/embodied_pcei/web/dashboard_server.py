"""
embodied_pcei/web/dashboard_server.py

High-performance FastAPI server providing real-time telemetry,
interactive controls, and live MJPEG visual simulation streaming
for the Embodied PCEI Ninja Warrior sandbox.
"""

import time
import asyncio
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import threading

from embodied_pcei.src.telemetry import GLOBAL_TELEMETRY

WEB_DIR = Path(__file__).resolve().parent
HTML_PATH = WEB_DIR / "index.html"

app = FastAPI(title="PCEI Embodied Sandbox Dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CameraRequest(BaseModel):
    camera: str


class NudgeRequest(BaseModel):
    x: float
    y: float


class GhostSwarmRequest(BaseModel):
    enabled: Optional[bool] = None
    alpha: Optional[float] = None


@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    with open(HTML_PATH, "r") as f:
        content = f.read()
    return HTMLResponse(
        content=content,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/api/status")
async def get_status():
    return JSONResponse(GLOBAL_TELEMETRY.get_status_dict())


@app.get("/api/history")
async def get_history():
    return JSONResponse(GLOBAL_TELEMETRY.get_history_dict())


@app.post("/api/camera")
async def set_camera(req: CameraRequest):
    GLOBAL_TELEMETRY.set_camera(req.camera)
    return {"status": "ok", "camera": req.camera}


@app.post("/api/ghost_swarm")
async def toggle_ghost_swarm(req: Optional[GhostSwarmRequest] = None):
    if req is not None:
        if req.enabled is not None:
            GLOBAL_TELEMETRY.set_ghost_swarm(req.enabled)
        if req.alpha is not None:
            GLOBAL_TELEMETRY.set_ghost_swarm_alpha(req.alpha)
        enabled = GLOBAL_TELEMETRY.ghost_swarm_enabled
    else:
        enabled = GLOBAL_TELEMETRY.toggle_ghost_swarm()
    return {
        "status": "ok",
        "enabled": enabled,
        "alpha": GLOBAL_TELEMETRY.ghost_swarm_alpha,
    }


@app.get("/api/ghost_swarm")
async def get_ghost_swarm():
    return {
        "enabled": GLOBAL_TELEMETRY.ghost_swarm_enabled,
        "alpha": GLOBAL_TELEMETRY.ghost_swarm_alpha,
    }


class ViewModeRequest(BaseModel):
    mode: str


@app.post("/api/view_mode")
async def set_view_mode(req: ViewModeRequest):
    GLOBAL_TELEMETRY.set_view_mode(req.mode)
    return {"status": "ok", "mode": req.mode}


class StreamRequest(BaseModel):
    stream: str


@app.post("/api/stream")
async def set_stream_post(req: StreamRequest):
    GLOBAL_TELEMETRY.set_stream_view(req.stream)
    return {"status": "ok", "stream": GLOBAL_TELEMETRY.active_stream_view}


@app.get("/api/set_stream")
async def set_stream_get(stream: str = "all"):
    GLOBAL_TELEMETRY.set_stream_view(stream)
    return {"status": "ok", "stream": GLOBAL_TELEMETRY.active_stream_view}


@app.post("/api/nudge")
async def trigger_nudge(req: NudgeRequest):
    GLOBAL_TELEMETRY.trigger_nudge(req.x, req.y)
    return {"status": "ok", "nudge": [req.x, req.y]}


async def mjpeg_frame_generator():
    """Generates continuous MJPEG multipart stream from the latest MuJoCo frame."""
    last_seq = -1
    while True:
        frame_bytes = GLOBAL_TELEMETRY.get_latest_frame()
        seq = GLOBAL_TELEMETRY.frame_seq
        if frame_bytes is not None and seq != last_seq:
            last_seq = seq
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
        await asyncio.sleep(0.03) # ~30 FPS polling limit


@app.get("/stream.mjpg")
async def stream_mjpg():
    return StreamingResponse(
        mjpeg_frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


def start_server_in_thread(host: str = "0.0.0.0", port: int = 8080) -> threading.Thread:
    """Spawns uvicorn server in a separate background daemon thread."""
    config = uvicorn.Config(app=app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    return t


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
