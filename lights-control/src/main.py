# lights-control/src/main.py
import threading
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager

from src.engine.compositor import Engine

# --- Data Models ---
class LayerModel(BaseModel):
    type: str
    color: list[int]
    faces: list[int] = [0, 1, 2, 3]
    h_min: float = 0.0
    h_max: float = 1.0

class SceneRequest(BaseModel):
    layers: list[LayerModel]

# --- Global Engine Instance ---
engine = Engine()

# --- Lifecycle Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Launch the rendering thread
    render_thread = threading.Thread(target=engine.start_loop, daemon=True)
    render_thread.start()
    print("🕯️ Infinite Candle Engine Started")
    yield
    # Shutdown
    engine.stop_loop()
    print("🛑 Engine Stopped")

app = FastAPI(lifespan=lifespan)

# --- Routes ---
@app.post("/scene")
async def set_scene(scene: SceneRequest):
    """
    Accepts a JSON payload to update the lights.
    Example:
    {
      "layers": [
        {"type": "solid", "color": [0, 0, 255], "h_max": 0.5},
        {"type": "solid", "color": [255, 0, 0], "h_min": 0.5}
      ]
    }
    """
    # Convert Pydantic models to dicts for the engine
    layer_data = [l.model_dump() for l in scene.layers]
    engine.update_layers(layer_data)
    return {"status": "Scene Updated", "layers_count": len(layer_data)}

@app.get("/status")
async def get_status():
    return {"running": engine.running, "led_count": engine.driver.count}

# Entry point for debugging
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)