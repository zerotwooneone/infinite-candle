# Infinite Candle 🕯️

<div align="center">
  <video src="candle.mp4" height="480" controls muted loop autoplay></video>
</div>

> **Status:** Developing new animations

**Infinite Candle** is a hardware/software platform for controlling a high-density, spirally-wrapped LED pillar. It transforms a standard 4-foot column into a Wi-Fi-enabled, algorithmic light display capable of rendering complex layered effects, weather-reactive patterns, and Cthulhu-summoning rituals.

The system runs on a **Raspberry Pi 3** using a custom Python rendering engine that abstracts physical pixels into a virtual 3D surface (Faces & Height).

## 📂 Repository Structure

This is a mono-repo organized into five domains:

| Directory | Purpose |
| :--- | :--- |
| **[`/hardware`](./hardware)** | **The Physical Realm.** Wiring diagrams, Bill of Materials (BOM), and electrical specifications for the 12V/5V split system. |
| **[`/system`](./system)** | **The Infrastructure.** Provisioning scripts (bash) to configure the Raspberry Pi OS, disable audio (for PWM timing), and manage systemd services. |
| **[`/lights-control`](./lights-control)** | **The Brain.** The Python application containing the FastAPI web server, the Scene Graph rendering engine, and the pixel mapping logic. |
| **[`/dashboard`](./dashboard)** | **The Face.** React + Vite + TypeScript control panel with a React Three Fiber digital twin that mirrors the live LED buffer over WebSocket. |
| **[`/offline-sim`](./offline-sim)** | **The Bakery.** Offline renderers (NumPy/CuPy) that bake complex effects into `.npy` clip files the engine can play back. |

> 🧩 **Writing a new animation?** Plugins live in `lights-control/src/plugins/builtin/` and need no dashboard or engine changes — see the **[Animation Plugin Guide](./plugin-guide.md)**.

---

## 🚀 Getting Started

### 1. Clone the Repo

```bash
git clone <repo-url> infinite-candle
cd infinite-candle
```

### 2. Dependencies

| Component | Requires |
| :--- | :--- |
| `lights-control` (backend) | Python 3.13+, [`uv`](https://docs.astral.sh/uv/) |
| `dashboard` (frontend) | Node.js 18+ / npm |
| `offline-sim` | Python 3.13+, `uv` (CuPy optional, GPU only) |

```bash
# Backend — creates .venv and installs FastAPI, NumPy, rpi_ws281x (linux only), etc.
cd lights-control
uv sync

# Frontend
cd ../dashboard
npm install
```

### 3. Run Locally (no Pi required)

The backend detects non-Pi environments and runs a mock driver, so the full stack works on any machine:

```bash
# Terminal 1 — API + engine (mock driver) on :8000
cd lights-control
.venv/bin/python src/main.py        # Windows: .venv\Scripts\python src/main.py

# Terminal 2 — dashboard dev server on :5173
cd dashboard
npm run dev
```

Open `http://localhost:5173`. The dashboard talks to the Pi's API by default — point it elsewhere with `VITE_CANDLE_HOST`:

```bash
# PowerShell
$env:VITE_CANDLE_HOST="10.0.135.7:8000"; npm run dev
# bash
VITE_CANDLE_HOST=10.0.135.7:8000 npm run dev
```

No Pi reachable? Toggle **Dev Mode** in the digital twin — it runs a local procedural simulator through the same rendering path.

### 4. The Hardware Build
Before deploying to a real pillar, the physical wiring must be correct to prevent magic smoke.
* **Voltage:** Hybrid 12V (LEDs) and 5V (Pi) system.
* **Critical:** Ensure **Common Ground** between the two power supplies.
* See **[hardware/README.md](./hardware/README.md)** for the wiring diagram.

### 5. Pi Provisioning (first time only)
The Pi requires OS-level hacks (disabling audio) to enable hardware PWM.

```bash
# On the Pi, from the repo root:
cd system/src

# 1. Root-level setup (disables audio, installs system deps)
sudo bash 01_system_prep.sh
sudo bash 02_sys_deps.sh

# 2. User-level setup (installs uv, syncs Python env)
bash 03_user_env.sh
```

---

## 🚢 Deploying to the Pi

The Pi runs `lights-control` as a systemd service named **`infinite-candle`**. FastAPI serves the API, the WebSocket stream, **and** the built dashboard from `dashboard/dist` — everything lives on port **8000**.

### Backend changes

```bash
# From your dev machine:
scp -r lights-control/src pi@<PI_IP>:~/infinite-candle/lights-control/
ssh pi@<PI_IP> "sudo systemctl restart infinite-candle"
```

### Dashboard changes

```bash
# Build the production bundle, then ship it:
cd dashboard
npm run build
scp -r dist pi@<PI_IP>:~/infinite-candle/dashboard/
ssh pi@<PI_IP> "sudo systemctl restart infinite-candle"
```

> **Note:** the static mount is registered at startup — restart the service after the *first* `dist` deploy. Subsequent asset-only updates serve immediately, but restarting is harmless and recommended.

### Verify

* Dashboard: `http://<PI_IP>:8000`
* API docs (Swagger): `http://<PI_IP>:8000/docs`
* Logs: `journalctl -u infinite-candle -f`

### Manual run (debugging on the Pi)

```bash
cd lights-control
sudo .venv/bin/python src/main.py   # sudo required for /dev/mem access
```

---

## 📐 Architecture

The Infinite Candle does not expose raw pixels to the outside world. Instead, it uses a **Scene Graph** architecture similar to game engines.

### The Virtual Mapper
Because the LEDs are wrapped in a spiral (approx. 26 wraps), physically adjacent pixels are not logically adjacent. The **Mapper** translates 3D coordinates into linear strip indices.
* **Input:** `{ Face: 0-3, Height: 0.0-1.0 }`
* **Output:** `Strip_Index (0-600)`

### The Rendering Pipeline
1.  **API Layer (FastAPI):** Receives high-level instructions (e.g., *"Add a Fire effect to the top 50% of Face 0"*).
2.  **Scene Manager:** Maintains a stack of active **Layers**.
3.  **Compositor:** Runs at ~60 FPS. It asks every active layer to calculate its color for the current frame, blends them (handling opacity/masking), and outputs the final buffer.
4.  **Driver:** Pushes the buffer to the hardware via `rpi_ws281x` — a single bulk DMA copy per frame (set `LED_BULK_WRITE = False` in `src/config.py` to fall back to per-pixel writes).

### Real-Time Interfaces

| Endpoint | Purpose |
| :--- | :--- |
| `POST /scene` | Replace the active layer stack (persisted to `scene.json`) |
| `GET /scene` | Authoritative active scene — clients rehydrate from this on load |
| `GET /mapping` | Normalized (azimuth, height) coordinate for each of the 600 LEDs |
| `GET /status` | Engine running state |
| `WS /ws/stream` | Live frame buffer — raw 1800-byte binary frames at ~15 FPS for the digital twin |

---

## 🛠️ Hardware Specs

* **Controller:** Raspberry Pi 3 Model B
* **LED Protocol:** WS2815 (12V, RGB, Backup Data Line)
* **Density:** ~600 LEDs (10 meters)
* **Geometry:**
    * Square Pillar: 3.75" width
    * Height: 48"
    * Wrap Density: ~26 wraps (1.85" vertical pitch)

## ⚠️ Safety & Constraints

* **Power:** The 12V rail can draw up to **10 Amps**. Ensure wiring is at least 18AWG for power injection lines.
* **Audio:** The Raspberry Pi onboard audio jack is **DISABLED** to free up the PWM generator for GPIO 18.
* **Root Access:** The rendering engine must run with `sudo` privileges to access `/dev/mem`.

---

*"Ph'nglui mglw'nafh Cthugha Fomalhaut n'gha-ghaa naf'lthagn."*