# Bill of Materials (BOM)

## Core Electronics

| Component | Specification | Quantity | Usage |
| :--- | :--- | :--- | :--- |
| **Controller** | Raspberry Pi 3 Model B | 1 | Runs Python control script. |
| **LED Strip** | WS2815 12V RGB (IP65) | 2 Reels | 12V, 4-wire (12V, DI, BI, GND). 60 LEDs/m. |
| **LED PSU** | 12V 10A DC Adapter | 1 | "Brick" style with barrel jack. Powers the LEDs. |
| **Pi PSU** | 5V 2.5A MicroUSB | 1 | Standard Pi wall adapter. Powers the Pi. |
| **Level Shifter** | 74AHCT125 | 1 | **Critical.** Boosts 3.3V GPIO to 5V Data. |

## Wiring & Connectors

| Component | Specification | Quantity | Usage |
| :--- | :--- | :--- | :--- |
| **Primary Wire** | 18 AWG (Silicone) | ~30ft | For 12V Power Injection lines (Red/Black). |
| **Data Wire** | 22 AWG (Silicone) | ~5ft | For Data signals (Green/Yellow). |
| **Power Adapter** | Female DC Barrel Jack | 1 | Adapts the 12V PSU plug to raw wires. |
| **Connectors** | Wago 221 (3-way) | ~5 | For splicing power injection points safely. |
| **Jumper Wires** | Female-to-Male | ~5 | For connecting Pi GPIO headers to breadboard/shifter. |

## Enclosure & Mounting

| Component | Specification | Usage |
| :--- | :--- | :--- |
| **Project Box** | Plastic (Waterproof) | Houses the Pi and Level Shifter. |
| **Adhesive** | 3M VHB Tape or Zip Ties | Secures the start/end of the strip to the pillar. |