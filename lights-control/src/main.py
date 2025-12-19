import logging
import threading

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

# 1. Import the schemas we defined
#    This brings in the "smart" SceneRequest that handles Stripes/Snow/etc.
from src.api.schemas import SceneRequest, SolidLayer

from src.engine.compositor import Engine
from src.scene_store import SceneStore

logger = logging.getLogger(__name__)

# --- Global Engine Instance ---
engine = Engine()
scene_store = SceneStore.default()

# --- Lifecycle Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        persisted_scene = scene_store.load()
        if persisted_scene is not None:
            engine.update_layers(persisted_scene.layers)
            logger.info("Loaded persisted scene from disk")
        else:
            engine.update_layers([])
            logger.warning("No persisted scene found (or load failed); bootstrapping with no active scene")
    except Exception:
        engine.update_layers([])
        logger.exception("Failed to load persisted scene; bootstrapping with no active scene")

    render_thread = threading.Thread(target=engine.start_loop, daemon=True)
    render_thread.start()
    print(" Infinite Candle Engine Started")
    yield
    engine.stop_loop()
    print(" Engine Stopped")

app = FastAPI(lifespan=lifespan)

# --- Routes ---

@app.post("/scene")
async def set_scene(scene: SceneRequest):
    # Uses the imported SceneRequest to validate layers
    engine.update_layers(scene.layers)
    try:
        scene_store.save(scene)
        return {"status": "Scene Updated", "persisted": True}
    except Exception:
        logger.exception("Scene updated but failed to persist scene JSON")
        return {"status": "Scene Updated", "persisted": False}

@app.get("/status")
async def get_status():
    return {"running": engine.running}

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
            <p>The engine is running.</p>
            <a href="/docs" style="color:#4af; font-size:1.5em;">Open Control Panel</a>
        </body>
    </html>
    """

# --- DEBUG ROUTES ---

@app.post("/debug/identify")
async def debug_identify():
    # Use the Object, not a dictionary
    green_layer = SolidLayer(type="solid", color=[0, 255, 0])
    engine.update_layers([green_layer])
    return {"status": "Highlighting bottom 5%"}

@app.post("/debug/identify/{face}")
async def debug_identify_face(face: int):
    blue_layer = SolidLayer(type="solid", color=[0, 0, 255])
    engine.update_layers([blue_layer])
    return {"status": f"Highlighting Face {face}"}

@app.post("/debug/{mode}")
async def debug_mode(mode: str):
    layers = []
    if mode == "white":
        layers.append(SolidLayer(type="solid", color=[255, 255, 255]))
    elif mode == "red":
        layers.append(SolidLayer(type="solid", color=[255, 0, 0]))
    elif mode == "green":
        layers.append(SolidLayer(type="solid", color=[0, 255, 0]))
    elif mode == "blue":
        layers.append(SolidLayer(type="solid", color=[0, 0, 255]))
    elif mode == "off":
        layers.append(SolidLayer(type="solid", color=[0, 0, 0]))
    else:
        return {"error": "Unknown mode"}

    engine.update_layers(layers)
    return {"status": f"Debug Mode: {mode}"}

if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    uvicorn.run(app, host="0.0.0.0", port=8000)