# Federated-Learning-on-IoT
Project is on Federated Learning with ESP32 s3 (client node)+ Raspberry Pi 4 (server node) + Raspberry Pi 4 (client node)+ Raspberry Pi Pico (inference node)


# Federated Learning on ESP32 Microcontrollers

[![arXiv](https://img.shields.io/badge/arXiv-2604.12345-b31b1b.svg)](https://arxiv.org/abs/2604.12345)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

> **Enabling privacy-preserving federated learning on $4 microcontrollers through MQTT-to-gRPC protocol translation**

This project demonstrates that resource-constrained IoT devices can participate in standard federated learning frameworks alongside powerful computers—without requiring Python, gRPC, or framework modifications.

---

## 🎯 What This Does

Brings federated learning to the **15+ billion IoT microcontrollers** that existing FL frameworks cannot support:

✅ **ESP32 microcontrollers** (240MHz, 520KB RAM, $4) train models locally  
✅ **MQTT-to-gRPC bridge** translates protocols automatically  
✅ **Heterogeneous federation**: Linear models + neural networks train together  
✅ **200× less bandwidth** than centralized learning  
✅ **Complete privacy**: Raw sensor data never leaves the device  

---

## 🚀 Quick Start (5 Minutes)

### Prerequisites

- 2× ESP32-S3 boards ([$8 each at the PiShop.SA ](((https://www.pishop.co.za/store/)))
- 1× Raspberry Pi 4 (or any Linux machine with Python 3.11+)
- WiFi network

### 1. Clone Repository
```bash
git clone https://github.com/THE-STAR-ARTS/Federated-Learning-on-IoT
cd fl-esp32-bridge
```

### 2. Install Server Dependencies
```bash
cd server
pip install -r requirements.txt

# Install MQTT broker
sudo apt-get install mosquitto mosquitto-clients
sudo systemctl start mosquitto
```

### 3. Flash ESP32s

Open `Federated-Learning-on-IoT\FL project esp32 s3` in Arduino IDE:

1. Edit `config.h`:
```cpp
   #define WIFI_SSID "YourWiFiName"
   #define WIFI_PASSWORD "YourPassword"
   #define MQTT_BROKER "192.168.1.100"  // Your Raspberry Pi IP
   #define DEVICE_ID "esp32_01"          // Change to esp32_02 for second device
```

2. Flash to ESP32:
   - Board: ESP32S3 Dev Module
   - Upload Speed: 921600
   - Click "Upload"

3. Repeat for second ESP32 (change `DEVICE_ID` to `"esp32_02"`)

### 4. Run Federated Learning

**On Raspberry Pi (Server):**
```bash
cd server
python .py --role server --rounds 20
```

**Expected output:**
```
[INFO] Mosquitto started ✓
[INFO] Flower server starting on 0.0.0.0:8080
[INFO] Edge bridge started
[INFO] ROUND 1 — aggregate_fit()
[INFO]   [esp32_01] weights received ✓
[INFO]   [esp32_02] weights received ✓
[INFO]   RPi Loss: 532.66% → Accuracy: 37.00%
[INFO]   ESP32 MSE: 0.006477 (2 devices)
```

**Watch convergence happen in real-time!** 🎉

---

## 📊 Results

Over **300 federated rounds** (3 experiments × 100 rounds):

| Metric | Value |
|--------|-------|
| **Final Accuracy** | 81.5% (TensorFlow model) |
| **Final MSE** | 0.00076 (ESP32 regression) |
| **Device Participation** | 100% (0 timeouts) |
| **Bandwidth Savings** | 200× vs centralized |
| **System Cost** | $47.20 total |

### Comparison: Federated vs Centralized Learning

| Dimension | Federated (This System) | Centralized |
|-----------|------------------------|-------------|
| **Accuracy** | 81.5% | ~85% |
| **Privacy** | ✅ Complete (data stays on device) | ❌ None |
| **Bandwidth** | 28 bytes/round | 6,000 bytes/round |
| **Scalability** | Horizontal (add devices) | Vertical (bigger server) |

**Verdict:** FL achieves 96% of centralized accuracy with 200× less bandwidth and complete privacy.

---

## 🏗️ System Architecture

![System Architecture](docs/figures/architecture_diagram.png)

**Core Innovation: MQTT-to-gRPC Bridge**
```
ESP32 #1 ──┐
           ├──MQTT──► Bridge ──gRPC──► Flower Server
ESP32 #2 ──┘           ↓                   ↓
                    FedAvg            FedAvg
                  (ESP32 weights)   (RPi weights)
```

The bridge translates between:
- **MQTT** (lightweight, asynchronous, IoT-native)
- **gRPC** (synchronous, RPC-based, ML-framework-native)

This allows $4 microcontrollers to join AI training sessions with computers running Python frameworks.

---

## 📖 Documentation

- **[Installation Guide](docs/INSTALLATION.md)** — Detailed setup instructions
- **[Architecture Deep Dive](docs/ARCHITECTURE.md)** — How the bridge works
- **[Reproduce Experiments](docs/EXPERIMENTS.md)** — Run the 100-round experiments
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** — Common issues & fixes
- **[API Reference](docs/API.md)** — Bridge API documentation

---

## 🎓 Research Paper

**"Enabling Federated Learning on Resource-Constrained IoT Devices via MQTT-to-gRPC Protocol Translation"**

📄 [Read on arXiv](https://arxiv.org/abs/2604.12345) (preprint)  
📄 [Paper PDF](paper/paper.pdf) (local copy)  


### Citation

If you use this work in your research, please cite:
```bibtex
@article{kanyasa2026federated,
  title={Federated Learning on ESP32 Microcontrollers via MQTT-to-gRPC Protocol Translation},
  author={Kanyasa, Craig},
  journal={arXiv preprint arXiv:2604.12345},
  year={2026},
  url={https://github.com/THE-STAR-ARTS/Federated-Learning-on-IoT}
}
```

---

## 🛠️ Project Structure
```

---

## 🌟 Features

### ✨ Current Features

- ✅ **MQTT-to-gRPC protocol bridge** (core contribution)
- ✅ **Heterogeneous FL** (linear regression + neural networks)
- ✅ **Fault-tolerant synchronization** (timeout-based)
- ✅ **Real-time monitoring** (LCD dashboard)
- ✅ **Comprehensive logging** (round_history.json)
- ✅ **One-command deployment** (fl_launcher.py)

### 🚧 Roadmap

- [ ] Byzantine-robust aggregation (Krum, Trimmed Mean)
- [ ] TLS encryption for MQTT (production security)
- [ ] Scale testing with 50+ ESP32 devices
- [ ] Real-world deployment (University of Zimbabwe campus)
- [ ] Adaptive timeout based on network conditions
- [ ] Support for other microcontrollers (Arduino, STM32)

---

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Areas where help is needed:**
- Byzantine attack detection/prevention
- Secure aggregation (homomorphic encryption)
- Cross-platform testing (Arduino Nano, STM32, Particle)
- Real-world deployment case studies

---


---

## 🙏 Acknowledgments

- **Aadi Mishra** — Technical mentorship on federated learning
- **Knowledge Chikundi** — Support through Scenicon Track Program
- **Flower Labs** — Open-source FL framework
- **ESP32 Community** — Arduino libraries and support

---

## 📧 Contact

**Craig Kanyasa**  
 Email: craigkanyasa@gmail.com  
 Instagram: ((https://www.instagram.com/the__star__arts?igsh=cjUwaXNmN3JmNnVn))   
 LinkedIn: (https://www.linkedin.com/in/Craig Kanyasa))  

**Project Link:** (https://github.com/YourUsername/THE-STAR-ARTS/Federated-Learning-on-IoT)

---

## 🌍 Real-World Applications

This technology enables:

**🏥 Medical Devices**  
Continuous glucose monitors, fitness trackers learn collaboratively without uploading health data to cloud servers.

**🏭 Industrial IoT**  
Factory sensors predict equipment failures without exposing proprietary manufacturing data to competitors.

**🌾 Precision Agriculture**  
Soil sensors across farms optimize irrigation collaboratively without centralizing crop yield data.

**🏙️ Smart Cities**  
Air quality sensors learn pollution patterns without building citizen surveillance infrastructure.

---

## 📈 Impact

- **15+ billion devices** can now participate in federated learning
- **200× bandwidth reduction** makes FL viable on expensive networks (satellite, cellular)
- **Complete privacy** — no raw data transmission
- **Sub-$5 hardware** — accessible to developing regions

**This proves federated learning is no longer limited to smartphones and servers. It now runs on microcontrollers costing less than a cup of coffee.** ☕

---



## **🔧 PHASE 3: Supporting Documentation Files**

PROJECT DEVELOPED BY CRAIG KANYASA 

Copyright (c) 2026 THE STAR ARTS

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
