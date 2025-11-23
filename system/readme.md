# System Configuration & Infrastructure

## Overview
This directory contains the infrastructure-as-code required to provision the Raspberry Pi to run the **Infinite Candle** display.

This project uses **uv** for ultra-fast, reliable Python dependency management and locking. Because WS2815 LEDs require precise hardware timing via DMA, we must also modify the OS configuration to disable audio interference and manage root permissions carefully.

## Directory Structure
* **`src/`**: Provisioning scripts to set up the environment.
* **`startup/`**: Systemd unit files for auto-start on boot.

## Prerequisites
* **OS:** Raspberry Pi OS (Lite version recommended).
* **User:** Scripts assume the standard user `pi`.
* **Network:** Online access to download `uv` and dependencies.

## Provisioning Runbook

### 1. System Prep (Audio Disable)
Run the configuration script to disable onboard audio. The Raspberry Pi shares hardware PWM registers between the 3.5mm audio jack and GPIO 18. Audio must be disabled for the LEDs to function without flickering.

```bash
cd system/src
sudo bash 01_system_prep.sh
# ⚠️ REBOOT REQUIRED AFTER THIS STEP
```

### 2. Install `uv`
We use `uv` instead of pip to manage the Python environment.

```bash
# Install uv via official script
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh

# Source the new path (so you can use 'uv' immediately)
source $HOME/.cargo/env
```

### 3. Sync Application Dependencies
Navigate to the application directory and let `uv` build the virtual environment from the `pyproject.toml` / `uv.lock`.

```bash
cd ~/infinite-candle/lights-control

# This creates the local .venv and installs rpi_ws281x
uv sync
```

### 4. Service Installation (Auto-Start)
Sets up the systemd daemon to run the display automatically on boot.

```bash
# Copy the service file
sudo cp ../system/startup/infinite-candle.service /etc/systemd/system/

# Reload systemd, enable, and start
sudo systemctl daemon-reload
sudo systemctl enable infinite-candle.service
sudo systemctl start infinite-candle.service
```

## Technical Constraints & Architecture

### ⚠️ The "Sudo + venv" Paradox
The `rpi_ws281x` library requires **root privileges** (`sudo`) to access the `/dev/mem` hardware registers for PWM control. However, `uv` installs dependencies into a user-level virtual environment (`.venv`).

If you simply run `sudo python main.py`, you are executing the **system global python**, which does *not* have the installed libraries.

**Correct Execution:**
You must point `sudo` explicitly to the Python binary *inside* the `uv` environment:

```bash
# Manual execution example
sudo /home/pi/infinite-candle/lights-control/.venv/bin/python src/main.py
```

### Systemd Configuration
The `infinite-candle.service` file is pre-configured to handle this pathing:
```ini
ExecStart=/home/pi/infinite-candle/lights-control/.venv/bin/python /home/pi/infinite-candle/lights-control/src/main.py
```

### Audio Conflict Verification
If lights are flickering or behaving erratically, verify that the audio drivers are unloaded:
```bash
lsmod | grep snd
# Should return empty. If text appears, audio is still on.
```

## Maintenance
To view the logs of the running service:
```bash
journalctl -u infinite-candle.service -f
```