import asyncio
import logging
import threading

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from fastapi.middleware.cors import CORSMiddleware

# 1. Import the schemas we defined
#    This brings in the "smart" SceneRequest that handles Stripes/Snow/etc.
from src.api.schemas import SceneRequest, SolidLayer

from src.engine.compositor import Engine
from src.scene_store import SceneStore
from src.plugins.registry import list_plugins_metadata

logger = logging.getLogger(__name__)

# --- Global Engine Instance ---
engine = Engine()
scene_store = SceneStore.default()

# Authoritative copy of the currently active scene (None until first set/load)
current_scene: SceneRequest | None = None


def apply_scene(scene: SceneRequest) -> None:
    """Single mutation point: swap the engine's layer stack and update
    the authoritative scene so GET /scene never goes stale."""
    global current_scene
    engine.update_layers(scene.layers)
    current_scene = scene

# --- Lifecycle Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        persisted_scene = scene_store.load()
        if persisted_scene is not None:
            apply_scene(persisted_scene)
            logger.info("Loaded persisted scene from disk")
        else:
            apply_scene(SceneRequest(layers=[]))
            logger.warning("No persisted scene found (or load failed); bootstrapping with no active scene")
    except Exception:
        apply_scene(SceneRequest(layers=[]))
        logger.exception("Failed to load persisted scene; bootstrapping with no active scene")

    render_thread = threading.Thread(target=engine.start_loop, daemon=True)
    render_thread.start()
    print(" Infinite Candle Engine Started")
    yield
    engine.stop_loop()
    print(" Engine Stopped")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Routes ---

@app.post("/scene")
async def set_scene(scene: SceneRequest):
    # Uses the imported SceneRequest to validate layers
    apply_scene(scene)
    try:
        scene_store.save(scene)
        return {"status": "Scene Updated", "persisted": True}
    except Exception:
        logger.exception("Scene updated but failed to persist scene JSON")
        return {"status": "Scene Updated", "persisted": False}

@app.get("/scene")
async def get_scene():
    """Return the authoritative active scene so clients can rehydrate."""
    if current_scene is None:
        return {"layers": []}
    return current_scene

@app.get("/plugins")
async def get_plugins():
    """Return metadata and editable parameter specs for all registered plugins."""
    return {"plugins": list_plugins_metadata()}

@app.patch("/scene/layers/{layer_ref}")
async def patch_scene_layer(layer_ref: str, updates: dict):
    """Live-update a running layer's parameters without resetting animation state."""
    ok = engine.patch_layer(layer_ref, updates)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Layer '{layer_ref}' not found")
    # Sync authoritative current_scene from engine's active configs
    global current_scene
    current_scene = SceneRequest(layers=[eff.config for eff in engine.active_effects])
    return {"status": "Layer Updated", "layer": layer_ref}

@app.get("/status")
async def get_status():
    return {"running": engine.running}

@app.get("/mapping")
async def get_mapping():
    return {
        "coords_x": engine.mapper.coords_x.tolist(),
        "coords_y": engine.mapper.coords_y.tolist()
    }

@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    """
    Stream the live frame buffer to the digital twin.
    Decimated to ~15 FPS and sent as raw binary (600*3 = 1800 bytes/frame)
    to keep Pi CPU and LAN bandwidth usage negligible.
    """
    await websocket.accept()
    try:
        while True:
            frame = engine.get_stream_frame()
            await websocket.send_bytes(frame)
            await asyncio.sleep(1.0 / 15.0)
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("WebSocket stream terminated unexpectedly")

# --- Dashboard Hosting ---
# If a production build of the React dashboard exists, serve it at /.
# Repo layout: <repo>/dashboard/dist relative to <repo>/lights-control/src/main.py
DASHBOARD_DIST = Path(__file__).resolve().parents[2] / "dashboard" / "dist"

if not DASHBOARD_DIST.exists():
    @app.get("/", response_class=HTMLResponse)
    async def root():
        return """
        <html>
            <head>
                <title>Infinite Candle</title>
                <style>body{font-family:sans-serif; text-align:center; padding:50px; background:#111; color:#eee;}</style>
            </head>
            <body>
                <h1>🕯️ Infinite Candle is Online</h1>
                <p>The engine is running. No dashboard build found at dashboard/dist.</p>
                <a href="/docs" style="color:#4af; font-size:1.5em;">Open Control Panel</a>
            </body>
        </html>
        """

# --- DEBUG ROUTES ---

@app.post("/debug/identify")
async def debug_identify():
    apply_scene(SceneRequest(layers=[SolidLayer(type="solid", color=[0, 255, 0])]))
    return {"status": "Highlighting bottom 5%"}

@app.post("/debug/identify/{face}")
async def debug_identify_face(face: int):
    apply_scene(SceneRequest(layers=[SolidLayer(type="solid", color=[0, 0, 255])]))
    return {"status": f"Highlighting Face {face}"}

@app.post("/debug/{mode}")
async def debug_mode(mode: str):
    colors = {
        "white": [255, 255, 255],
        "red": [255, 0, 0],
        "green": [0, 255, 0],
        "blue": [0, 0, 255],
        "off": [0, 0, 0],
    }
    color = colors.get(mode)
    if color is None:
        return {"error": "Unknown mode"}

    apply_scene(SceneRequest(layers=[SolidLayer(type="solid", color=color)]))
    return {"status": f"Debug Mode: {mode}"}

# Mount the dashboard build LAST — mounts are fallthrough, but keeping it
# after every API route guarantees /scene, /mapping, /ws/stream, etc. win.
if DASHBOARD_DIST.exists():
    app.mount("/", StaticFiles(directory=DASHBOARD_DIST, html=True), name="dashboard")

if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    uvicorn.run(app, host="0.0.0.0", port=8000)