# Infinite Candle 🕯️

> **Status:** Active Development

**Infinite Candle** is a hardware/software platform for controlling a high-density, spirally-wrapped LED pillar. It transforms a standard 4-foot column into a Wi-Fi-enabled, algorithmic light display capable of rendering complex layered effects, weather-reactive patterns, and Cthulhu-summoning rituals.

The system runs on a **Raspberry Pi 3** using a custom Python rendering engine that abstracts physical pixels into a virtual 3D surface (Faces & Height).

## 📂 Repository Structure

This is a mono-repo organized into three distinct domains:

| Directory | Purpose |
| :--- | :--- |
| **[`/hardware`](./hardware)** | **The Physical Realm.** Wiring diagrams, Bill of Materials (BOM), and electrical specifications for the 12V/5V split system. |
| **[`/system`](./system)** | **The Infrastructure.** Provisioning scripts (bash) to configure the Raspberry Pi OS, disable audio (for PWM timing), and manage systemd services. |
| **[`/lights-control`](./lights-control)** | **The Brain.** The Python application containing the FastAPI web server, the Scene Graph rendering engine, and the pixel mapping logic. |

---

## 🚀 Quick Start

### 1. The Hardware Build
Before writing code, the physical pillar must be wired correctly to prevent magic smoke.
* **Voltage:** Hybrid 12V (LEDs) and 5V (Pi) system.
* **Critical:** Ensure **Common Ground** between the two power supplies.
* See **[hardware/README.md](./hardware/README.md)** for the wiring diagram.

### 2. System Provisioning
This project uses **`uv`** for dependency management and requires specific OS-level hacks (disabling audio) to enable hardware PWM.

```bash
# From the Pi:
cd system/src

# 1. Root-level Setup (Disables Audio, Install System Deps)
sudo bash 01_system_prep.sh
sudo bash 02_sys_deps.sh

# 2. User-level Setup (Installs uv, Syncs Python Env)
bash 03_user_env.sh
```

### 3. Running the Engine
Once the system is prepped, launch the control server. Note that we must use `sudo` to access hardware, but point to the `uv` virtual environment python binary.

```bash
cd lights-control
sudo .venv/bin/python src/main.py
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
4.  **Driver:** Pushes the buffer to the hardware via `rpi_ws281x`.

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