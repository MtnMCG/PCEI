import sys
import os
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(os.path.abspath(__file__)).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

from dashboard.api.model_stats import router as model_stats_router
from dashboard.api.training import router as training_router
from dashboard.api.websocket import router as websocket_router

app = FastAPI(title="PCEI Command Center")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store global paths
app.state.project_root = PROJECT_ROOT

# Include API routers
app.include_router(model_stats_router, prefix="/api")
app.include_router(training_router, prefix="/api")
app.include_router(websocket_router)

# Mount static files
static_dir = PROJECT_ROOT / "dashboard" / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("dashboard.server:app", host="0.0.0.0", port=8000, reload=True)
