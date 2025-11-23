# Hardware Documentation: Infinite Candle

## Overview
This directory documents the physical construction, wiring, and electrical specifications for the **Infinite Candle** project. The system uses a Raspberry Pi 3 to drive approximately 10 meters of WS2815 (12V) individually addressable LEDs wrapped around a 4-foot pillar.

## System Specifications
* **Voltage:** Split System (5V Logic/Computer, 12V LEDs).
* **Total LED Count:** ~600 LEDs (2x 5m reels @ 60 LEDs/m).
* **Strip Type:** WS2815 (12V, RGB, Backup Data Line).
* **Logic Voltage:** 3.3V (Pi) stepped up to 5V (Data Line).
* **Est. Max Power Draw:** ~120W (White @ 100%).
* **Safe Operating Current:** Limited via software to 9A (9000mA).

## Parts List
For specific component part numbers, links, and quantities, please refer to the [Bill of Materials](./bom.md).

## Wiring Guide (Dual Power Supply)

### ⚠️ CRITICAL WARNING: Common Ground
Because two separate wall outlets are used, the **Ground (GND)** of the Raspberry Pi and the **Ground (GND)** of the 12V LED Power Supply **MUST** be connected. Without this reference wire, data signals will fail, and lights may flash randomly.

### 1. Power Distribution
* **Raspberry Pi:** Powered independently via MicroUSB port (Standard Wall Adapter).
* **LED Strips:** Powered via 12V 10A Brick Adapter.
    * **Injection Points:** Power is injected at the **Start** (Strip 1) and the **End** (Strip 2) to prevent voltage drop (pinking).
    * **Strip Joints:** Strip 1 and Strip 2 are connected via JST connectors (Power + Data + Ground).

### 2. Signal Path (Data)
The Raspberry Pi outputs a 3.3V PWM signal which is too weak/noisy for a long run of 12V LEDs. It is passed through a Level Shifter.

**Pin Mapping:**

| Signal | Source (Pi) | Intermediate (74AHCT125) | Destination (LED Strip) |
| :--- | :--- | :--- | :--- |
| **PWM Data** | GPIO 18 (Physical 12) | Pin 2 (1A) -> Pin 3 (1Y) | **Green Wire (DI)** |
| **Reference** | Physical 6 (GND) | Pin 7 (GND) | **White Wire (GND)** |
| **HV Ref** | N/A | Pin 14 (VCC) -> 5V (Pi Pin 2) | N/A |
| **Enable** | N/A | Pin 1 (1OE) -> GND | N/A (Activates Shifter) |

### 3. The "Backup" Wire (Blue/BI)
The WS2815 has a redundant data line.
* **At the Start:** Connect the Blue Wire (BI) to **GND**.
* **Between Strips:** Connect Blue (Strip 1) to Blue (Strip 2).

## Installation Notes
1.  **Audio Disable:** The Pi's onboard audio must be disabled in `/boot/config.txt` (`dtparam=audio=off`) or the LEDs will flicker due to PWM conflict.
2.  **Heat Management:** The 12V PSU may get warm. Ensure it is not enclosed in an airtight box.
3.  **Coil Warning:** Do not power the LED strips while they are still coiled on the reel. They will overheat and melt the silicone within minutes. Unspool before testing.