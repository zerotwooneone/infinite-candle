import threading
from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.responses import HTMLResponse

# 1. Import the schemas we defined
from src.api.schemas import SceneRequest, SolidLayer, ChaseLayer, SnowLayer

from src.engine.compositor import Engine

# --- Global Engine Instance ---
engine = Engine()

# --- Lifecycle Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
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
    # This now receives a list of Typed Objects (SolidLayer, StripesLayer, etc.)
    engine.update_layers(scene.layers)
    return {"status": "Scene Updated"}

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

# --- DEBUG ROUTES (Updated to use Objects) ---

@app.post("/debug/identify")
async def debug_identify():
    """
    Runs a slow animation to help identify strip direction.
    """
    # We must send Objects now, not dictionaries!
    # Note: We haven't implemented masking in the new SolidEffect yet, 
    # so this will light up the WHOLE pillar green for now unless we add masking logic back.

    green_layer = SolidLayer(type="solid", color=[0, 255, 0])

    engine.update_layers([green_layer])
    return {"status": "Green Layer Sent (Masking momentarily disabled in V2)"}

@app.post("/debug/identify/{face}")
async def debug_identify_face(face: int):
    # In V2 architecture, we need to handle faces in the Effect logic.
    # Sending a solid blue layer for now.
    blue_layer = SolidLayer(type="solid", color=[0, 0, 255])
    engine.update_layers([blue_layer])
    return {"status": f"Blue Layer Sent (Face {face})"}

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
    uvicorn.run(app, host="0.0.0.0", port=8000)