# lights-control/src/main.py
import threading
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from fastapi.responses import HTMLResponse

from src.engine.compositor import Engine

# --- Data Models ---
class LayerModel(BaseModel):
    model_config = ConfigDict(extra='allow')
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
@app.post("/debug/identify")
async def debug_identify():
    """
    Runs a slow animation to help identify strip direction.
    """
    # This is a 'special' blocking test that overrides the engine loop briefly
    # In a real system we'd make this a 'Layer', but for quick testing:

    green = [0, 255, 0]
    off = [0, 0, 0]

    # Flash first 10 pixels to identify the START
    layers = [{"type": "solid", "color": off}] # clear
    engine.update_layers(layers)

    # We manually hijack the buffer for a second (Dirty but effective for identifying)
    # Ideally, you just create a "Chase" effect in the engine.
    # For now, let's just set the bottom 5% to Green
    layers = [{
        "type": "solid",
        "color": green,
        "h_min": 0.0,
        "h_max": 0.05
    }]
    engine.update_layers(layers)

    return {"status": "Highlighting bottom 5% (The Start)"}

@app.post("/debug/identify/{face}")
async def debug_identify_face(face: int):
    """
    Lights up a specific side of the pillar (0, 1, 2, or 3)
    """
    layers = [{
        "type": "solid",
        "color": [0, 0, 255],  # Blue
        "faces": [face],       # Only this specific face
        "h_min": 0.0,
        "h_max": 1.0
    }]
    engine.update_layers(layers)
    return {"status": f"Highlighting Face {face}"}

@app.post("/debug/{mode}")
async def debug_mode(mode: str):
    """
    Modes: 'white', 'red', 'green', 'blue', 'off'
    """
    layers = []

    if mode == "white":
        # CAUTION: Max Power Draw
        layers.append({"type": "solid", "color": [255, 255, 255]})

    elif mode == "red":
        layers.append({"type": "solid", "color": [255, 0, 0]})

    elif mode == "green":
        layers.append({"type": "solid", "color": [0, 255, 0]})

    elif mode == "blue":
        layers.append({"type": "solid", "color": [0, 0, 255]})

    elif mode == "off":
        # Send empty layer list or black layer
        layers.append({"type": "solid", "color": [0, 0, 0]})

    else:
        return {"error": "Unknown mode. Use: white, red, green, blue, off"}

    # Push to engine
    engine.update_layers(layers)
    return {"status": f"Debug Mode: {mode}"}

# Entry point for debugging
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)