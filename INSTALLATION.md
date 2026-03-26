# Installation Guide

Complete step-by-step setup instructions for the FL system.

## Prerequisites

### Hardware
- 2× ESP32-S3 development boards ([$4 each](https://www.aliexpress.com/...))
- 1× Raspberry Pi 4 (4GB RAM recommended) OR any Linux machine
- MicroUSB cables for ESP32 programming
- WiFi network (2.4GHz, ESP32 compatible)

### Software
- **Server:** Linux (Ubuntu 22.04 recommended), Python 3.11+
- **ESP32:** Arduino IDE 2.0+
- **Optional:** Raspberry Pi OS for RPi client

---

## Step 1: Server Setup (Raspberry Pi or Linux PC)

### 1.1 Install Python Dependencies
```bash
# Clone repository
git clone https://github.com/YourUsername/fl-esp32-bridge.git
cd fl-esp32-bridge/server

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 1.2 Install MQTT Broker
```bash
# Install Mosquitto
sudo apt-get update
sudo apt-get install mosquitto mosquitto-clients

# Start broker
sudo systemctl start mosquitto
sudo systemctl enable mosquitto  # Auto-start on boot

# Verify it's running
sudo systemctl status mosquitto
# Should show "active (running)"
```

### 1.3 Configure Firewall (If Applicable)
```bash
# Allow MQTT port
sudo ufw allow 1883/tcp

# Allow Flower gRPC port
sudo ufw allow 8080/tcp
```

### 1.4 Get Server IP Address
```bash
hostname -I
# Example output: 192.168.1.100
# Note this IP — you'll need it for ESP32 configuration
```

---

## Step 2: ESP32 Setup

### 2.1 Install Arduino IDE

1. Download from [arduino.cc/en/software](https://www.arduino.cc/en/software)
2. Install and launch Arduino IDE
              OR
1. use PlatformIO on VScode
2. Download the extension on the extension market
3. use the platform.ini in FL project esp32 s3
   

### 2.2 Install ESP32 Board Support

1. Go to **File → Preferences**
2. In "Additional Board Manager URLs", add:
```
   https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
```
3. Go to **Tools → Board → Boards Manager**
4. Search for "esp32"
5. Install "esp32 by Espressif Systems" (version 2.0.14 or later)

### 2.3 Install Required Libraries

Go to **Sketch → Include Library → Manage Libraries**, search and install:
- `PubSubClient` by Nick O'Leary (version 2.8+)
- `ArduinoJson` by Benoit Blanchon (version 6.21+)

### 2.4 Configure ESP32 #1

1. Open `esp32_client/esp32_fl_client/esp32_fl_client.ino` in Arduino IDE

2. Open `config.h` tab and edit:
```cpp
   // WiFi credentials
   #define WIFI_SSID "YourWiFiName"
   #define WIFI_PASSWORD "YourWiFiPassword"
   
   // MQTT broker (your server IP from Step 1.4)
   #define MQTT_BROKER "192.168.1.100"  // CHANGE THIS
   #define MQTT_PORT 1883
   
   // Unique device ID
   #define DEVICE_ID "esp32_01"  // First device
```

3. Connect ESP32 via USB

4. Select board: **Tools → Board → ESP32 Arduino → ESP32S3 Dev Module**

5. Select port: **Tools → Port → /dev/ttyUSB0** (Linux) or **COM3** (Windows)

6. Click **Upload** (arrow button)

7. Open **Tools → Serial Monitor** (115200 baud)

8. You should see:
```
   [WiFi] Connected to YourWiFiName
   [WiFi] IP address: 192.168.1.105
   [MQTT] Connected to broker
   [FL] Ready — waiting for global weights
```

### 2.5 Configure ESP32 #2

Repeat Step 2.4 with **one change**:
- In `config.h`, set `#define DEVICE_ID "esp32_02"`  // Second device

Upload and verify both ESP32s are connected via serial monitor.

---

## Step 3: Run Federated Learning

### 3.1 Start Server

On Raspberry Pi / Linux PC:
```bash
cd server
source venv/bin/activate  # If using virtual environment

python fl_launcher.py --role server --rounds 20
```

**Expected output:**
```
============================================================
  FL SYSTEM LAUNCHER — SERVER MODE
============================================================
  Server IP: 192.168.1.100
  Rounds:    20
  Clients:   min 1
============================================================

[INFO] ✓ Mosquitto already running
[INFO] Flower server starting on 0.0.0.0:8080
[INFO] Edge bridge started
[INFO] Waiting for clients to connect...
```

### 3.2 Observe Training

Watch the server terminal for rounds:
```
[INFO] ROUND 1 — aggregate_fit()
[INFO]   RPi clients: 0 responded
[INFO]   [edge_bridge] Waiting for ESP32 devices...
[INFO]   [esp32_01] weights received ✓
[INFO]   [esp32_02] weights received ✓

  ── Model B (RPi TF) ──────────
     (no RPi client connected)

  ── Model A (ESP32 linear) ────
     avg MSE  : 0.006477
     devices  : 2

[INFO] Round 1/20 complete
```

Watch ESP32 serial monitors:
```
[FL] Global weights received ✓ — starting new round
Step   100 │ MSE: 0.008234 │ RMSE: 0.090742
Step   200 │ MSE: 0.007891 │ RMSE: 0.088826
...
Step   500 │ MSE: 0.006477 │ RMSE: 0.080480
╔══════════════════════════════════════════╗
║ ROUND   1 PUBLISHED TO MQTT              ║
╚══════════════════════════════════════════╝
```

### 3.3 (Optional) Add Raspberry Pi TensorFlow Client

On a second Raspberry Pi:
```bash
cd rpi_client
source venv/bin/activate
pip install -r requirements.txt

# Set server IP (from Step 1.4)
export FL_SERVER_IP=192.168.1.100

python rpi_tf_client.py
```

Now you'll see both ESP32 and RPi training in the server logs.

---

## Step 4: Monitor Results

Results are logged to `server/round_history.json`:
```bash
cat server/round_history.json
```

You can also visualize with:
```bash
python scripts/plot_convergence.py --input round_history.json
```

---

## Troubleshooting

### ESP32 Can't Connect to WiFi
- Check SSID/password in `config.h`
- Ensure WiFi is 2.4GHz (ESP32 doesn't support 5GHz)
- Move ESP32 closer to router

### ESP32 Can't Connect to MQTT
- Verify MQTT broker is running: `sudo systemctl status mosquitto`
- Check firewall: `sudo ufw allow 1883/tcp`
- Verify MQTT_BROKER IP is correct (server IP, not ESP32 IP)
- Ping server from ESP32 network: `ping 192.168.1.100`

### Timeout Errors
- Increase timeout in `server/edge_bridge.py`:
```python
  arrived = self.buffers[device_id].wait_for_update(timeout=60.0)  # Increase to 60s
```

### "No module named 'flwr'"
- Make sure virtual environment is activated: `source venv/bin/activate`
- Reinstall: `pip install -r requirements.txt`

For more issues, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

---

## Next Steps

- Run experiments: [EXPERIMENTS.md](EXPERIMENTS.md)
- Read architecture docs: [ARCHITECTURE.md](ARCHITECTURE.md)
- Try examples: `examples/01_basic_setup/`

---

**Questions? Open an issue on GitHub or contact craigkanyasa@gmail.com or whatsapp - 0779461598**
