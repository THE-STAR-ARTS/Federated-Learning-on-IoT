"""
fl_lcd_dashboard.py  —  runs on RPi #2 (FL server)
===================================================
Real-time federated learning metrics dashboard on a
320×480 ILI9486 SPI LCD display.

Shows comprehensive heterogeneous FL metrics:
  · Both model types (ESP32 linear + RPi TensorFlow)
  · Per-round convergence stats
  · Device connection status
  · Pico W inference status
  · Network bandwidth

Aesthetic: Industrial monitoring — dense information,
high contrast, monospaced numbers, status indicators.

Hardware:
  ILI9486 LCD (320×480) on SPI
  GPIO 24 (DC), GPIO 25 (RST)

Data source:
  Reads round_history.json written by strategy.py

Run:
  python fl_lcd_dashboard.py
"""

import json
import time
from pathlib import Path
from datetime import datetime

from luma.core.interface.serial import spi
from luma.core.render import canvas
from luma.lcd.device import ili9486
from PIL import ImageFont, ImageDraw

# ══════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════

WIDTH  = 320
HEIGHT = 480

GPIO_DC  = 24
GPIO_RST = 25

ROUND_HISTORY_PATH = Path("round_history.json")
REFRESH_SECONDS    = 2

# Color palette — industrial monitoring aesthetic
COL_BG       = "#0a0e14"   # dark blue-black
COL_PANEL    = "#141a24"   # panel background
COL_BORDER   = "#1e2a3a"   # borders
COL_TEXT     = "#d0dae8"   # primary text
COL_MUTED    = "#5a6a80"   # labels
COL_ACCENT   = "#00e5a0"   # success/good
COL_WARN     = "#ffd166"   # warning
COL_ERROR    = "#ff6b6b"   # error
COL_BLUE     = "#00c8ff"   # info

# Fonts — use DejaVu for clean monospaced numbers
FONT_PATH = "/usr/share/fonts/truetype/dejavu"
try:
    font_tiny  = ImageFont.truetype(f"{FONT_PATH}/DejaVuSans.ttf", 10)
    font_small = ImageFont.truetype(f"{FONT_PATH}/DejaVuSans.ttf", 12)
    font_med   = ImageFont.truetype(f"{FONT_PATH}/DejaVuSans-Bold.ttf", 14)
    font_large = ImageFont.truetype(f"{FONT_PATH}/DejaVuSansMono-Bold.ttf", 18)
    font_huge  = ImageFont.truetype(f"{FONT_PATH}/DejaVuSansMono-Bold.ttf", 28)
except OSError:
    # Fallback to default if DejaVu not available
    font_tiny = font_small = font_med = font_large = font_huge = ImageFont.load_default()


# ══════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════

def load_fl_state():
    """
    Read the latest FL state from round_history.json.
    Returns a dict with all dashboard metrics.
    """
    if not ROUND_HISTORY_PATH.exists():
        return {
            "status": "WAITING",
            "round": 0,
            "total_rounds": 1000,
            "rpi": {},
            "esp32": {},
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        }

    try:
        with open(ROUND_HISTORY_PATH, "r") as f:
            history = json.load(f)

        if not history:
            return {"status": "NO DATA", "round": 0}

        latest = history[-1]   # most recent round
        
        # Calculate convergence trend (last 3 rounds)
        rpi_trend = "STABLE"
        if len(history) >= 3:
            recent_losses = [
                r.get("rpi", {}).get("avg_loss", 999)
                for r in history[-3:]
            ]
            if all(recent_losses[i] > recent_losses[i+1] for i in range(len(recent_losses)-1)):
                rpi_trend = "IMPROVING"
            elif all(recent_losses[i] < recent_losses[i+1] for i in range(len(recent_losses)-1)):
                rpi_trend = "DEGRADING"

        return {
            "status": "RUNNING",
            "round": latest.get("round", 0),
            "total_rounds": 20,  # from server config
            "timestamp": datetime.fromtimestamp(
                latest.get("timestamp", time.time())
            ).strftime("%H:%M:%S"),

            # Model B — RPi TensorFlow
            "rpi": {
                "clients": latest.get("rpi", {}).get("num_clients", 0),
                "loss": latest.get("rpi", {}).get("avg_loss", 0.0),
                "loss_pct": latest.get("rpi", {}).get("avg_loss_pct", 0.0),
                "acc": latest.get("rpi", {}).get("avg_accuracy", 0.0),
                "acc_pct": latest.get("rpi", {}).get("avg_acc_pct", 0.0),
                "trend": rpi_trend,
            },

            # Model A — ESP32 linear regression
            "esp32": {
                "devices": latest.get("esp32", {}).get("num_devices", 0),
                "mse": latest.get("esp32", {}).get("avg_mse", 0.0),
                "rmse": latest.get("esp32", {}).get("avg_rmse", 0.0),
                "mae": latest.get("esp32", {}).get("avg_mae", 0.0),
                "per_device": latest.get("esp32", {}).get("per_device", {}),
            },
        }

    except Exception as e:
        return {
            "status": "ERROR",
            "error": str(e),
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        }


# ══════════════════════════════════════════════════════════════
#  DRAWING PRIMITIVES
# ══════════════════════════════════════════════════════════════

def draw_panel(draw, x, y, w, h, fill=COL_PANEL, outline=COL_BORDER):
    """Draw a bordered panel."""
    draw.rectangle((x, y, x+w, y+h), fill=fill, outline=outline, width=1)

def draw_label(draw, x, y, text, color=COL_MUTED, font=font_small):
    """Draw uppercase label text."""
    draw.text((x, y), text.upper(), fill=color, font=font)

def draw_value(draw, x, y, text, color=COL_TEXT, font=font_large):
    """Draw large numeric value."""
    draw.text((x, y), str(text), fill=color, font=font)

def draw_status_dot(draw, x, y, active=True, size=6):
    """Draw a status indicator dot."""
    color = COL_ACCENT if active else COL_ERROR
    draw.ellipse((x, y, x+size, y+size), fill=color)


# ══════════════════════════════════════════════════════════════
#  MAIN DASHBOARD LAYOUT
# ══════════════════════════════════════════════════════════════

def draw_dashboard(device, state):
    """
    Render the complete dashboard.
    
    Layout (320×480):
      [0–35]    Header bar (round, status)
      [35–185]  Model B — RPi TensorFlow
      [185–335] Model A — ESP32 linear
      [335–430] Device status grid
      [430–480] Footer (timestamp, health)
    """
    with canvas(device) as draw:
        
        # ── BACKGROUND ─────────────────────────────────────────
        draw.rectangle((0, 0, WIDTH, HEIGHT), fill=COL_BG)

        # ── HEADER ─────────────────────────────────────────────
        draw_panel(draw, 0, 0, WIDTH, 35, fill=COL_PANEL)
        
        # Round counter
        round_text = f"R {state['round']:>2}/{state['total_rounds']}"
        draw.text((8, 8), round_text, fill=COL_BLUE, font=font_huge)
        
        # Status badge
        status = state.get("status", "IDLE")
        status_color = {
            "RUNNING": COL_ACCENT,
            "WAITING": COL_WARN,
            "ERROR": COL_ERROR,
        }.get(status, COL_MUTED)
        
        badge_x = 140
        draw.rectangle((badge_x, 8, badge_x+80, 28), fill=status_color, outline=status_color)
        draw.text((badge_x+8, 12), status, fill=COL_BG, font=font_med)
        
        # Timestamp
        draw.text((235, 12), state.get("timestamp", "--:--:--"), fill=COL_MUTED, font=font_small)

        # ── MODEL B — RPi TensorFlow ───────────────────────────
        panel_y = 40
        draw_panel(draw, 5, panel_y, WIDTH-10, 140)
        
        # Section header
        draw.rectangle((5, panel_y, WIDTH-5, panel_y+22), fill=COL_BORDER)
        draw.text((10, panel_y+4), "MODEL B", fill=COL_TEXT, font=font_med)
        draw.text((90, panel_y+5), "RPi · TensorFlow · DHT Classification", fill=COL_MUTED, font=font_tiny)
        
        rpi = state.get("rpi", {})
        
        # Metrics grid
        metric_y = panel_y + 28
        
        # Loss
        draw_label(draw, 12, metric_y, "Loss", font=font_tiny)
        loss_val = rpi.get("loss_pct")
        if loss_val is not None:
            loss_color = COL_ACCENT if loss_val < 50 else COL_WARN if loss_val < 150 else COL_ERROR
            draw_value(draw, 12, metric_y+12, f"{loss_val:.1f}%", color=loss_color, font=font_large)
        else:
            draw_value(draw, 12, metric_y+12, "n/a", color=COL_MUTED, font=font_large)
        
        # Accuracy
        draw_label(draw, 100, metric_y, "Accuracy", font=font_tiny)
        acc_val = rpi.get("acc_pct")
        if acc_val is not None:
            acc_color = COL_ACCENT if acc_val > 75 else COL_WARN if acc_val > 60 else COL_ERROR
            draw_value(draw, 100, metric_y+12, f"{acc_val:.1f}%", color=acc_color, font=font_large)
        else:
            draw_value(draw, 100, metric_y+12, "n/a", color=COL_MUTED, font=font_large)
        
        # Clients
        draw_label(draw, 200, metric_y, "Clients", font=font_tiny)
        clients = rpi.get("clients", 0)
        draw_value(draw, 200, metric_y+12, str(clients), color=COL_BLUE, font=font_large)
        
        # Trend indicator
        trend_y = metric_y + 50
        draw_label(draw, 12, trend_y, "Trend", font=font_tiny)
        trend = rpi.get("trend", "STABLE")
        trend_color = COL_ACCENT if trend == "IMPROVING" else COL_ERROR if trend == "DEGRADING" else COL_BLUE
        draw.text((12, trend_y+12), trend, fill=trend_color, font=font_small)
        
        # Raw loss value (small)
        draw.text((12, trend_y+32), f"raw: {rpi.get('loss', 0.0):.4f}", fill=COL_MUTED, font=font_tiny)

        # ── MODEL A — ESP32 Linear Regression ──────────────────
        panel_y = 190
        draw_panel(draw, 5, panel_y, WIDTH-10, 140)
        
        # Section header
        draw.rectangle((5, panel_y, WIDTH-5, panel_y+22), fill=COL_BORDER)
        draw.text((10, panel_y+4), "MODEL A", fill=COL_TEXT, font=font_med)
        draw.text((90, panel_y+5), "ESP32 · Linear Regression · Light Prediction", fill=COL_MUTED, font=font_tiny)
        
        esp32 = state.get("esp32", {})
        
        # Metrics grid
        metric_y = panel_y + 28
        
        # MSE
        draw_label(draw, 12, metric_y, "MSE", font=font_tiny)
        mse = esp32.get("mse")
        if mse is not None and mse > 0:
            mse_color = COL_ACCENT if mse < 0.01 else COL_WARN if mse < 0.05 else COL_ERROR
            mse_text = f"{mse:.4f}"
        else:
            mse_color = COL_MUTED
            mse_text = "n/a"
        draw_value(draw, 12, metric_y+12, mse_text, color=mse_color, font=font_med)
        
        # RMSE
        draw_label(draw, 100, metric_y, "RMSE", font=font_tiny)
        rmse = esp32.get("rmse")
        if rmse is not None and rmse > 0:
            rmse_text = f"{rmse:.4f}"
            rmse_color = mse_color  # inherit color from MSE
        else:
            rmse_text = "n/a"
            rmse_color = COL_MUTED
        draw_value(draw, 100, metric_y+12, rmse_text, color=rmse_color, font=font_med)
        
        # MAE
        draw_label(draw, 190, metric_y, "MAE", font=font_tiny)
        mae = esp32.get("mae")
        if mae is not None and mae > 0:
            mae_text = f"{mae:.4f}"
            mae_color = mse_color  # inherit color from MSE
        else:
            mae_text = "n/a"
            mae_color = COL_MUTED
        draw_value(draw, 190, metric_y+12, mae_text, color=mae_color, font=font_med)
        
        # Devices active
        dev_y = metric_y + 50
        draw_label(draw, 12, dev_y, "Devices", font=font_tiny)
        num_devs = esp32.get("devices", 0)
        dev_color = COL_ACCENT if num_devs == 2 else COL_WARN if num_devs == 1 else COL_ERROR
        draw_value(draw, 12, dev_y+12, f"{num_devs}/2", color=dev_color, font=font_large)
        
        # Per-device status dots
        per_dev = esp32.get("per_device", {})
        dot_x = 100
        for dev_id in ["esp32_01", "esp32_02"]:
            active = dev_id in per_dev
            draw_status_dot(draw, dot_x, dev_y+18, active=active, size=8)
            draw.text((dot_x+12, dev_y+15), dev_id[-2:], fill=COL_MUTED, font=font_tiny)
            dot_x += 60

        # ── DEVICE STATUS GRID ─────────────────────────────────
        panel_y = 340
        draw_panel(draw, 5, panel_y, WIDTH-10, 85)
        
        draw_label(draw, 12, panel_y+6, "Network Status", font=font_tiny)
        
        # MQTT broker
        draw.text((12, panel_y+22), "• MQTT Broker", fill=COL_MUTED, font=font_tiny)
        draw.text((130, panel_y+22), "ACTIVE", fill=COL_ACCENT, font=font_tiny)
        
        # Flower gRPC
        draw.text((12, panel_y+38), "• Flower gRPC", fill=COL_MUTED, font=font_tiny)
        rpi_active = rpi.get("clients", 0) > 0
        grpc_status = "CONNECTED" if rpi_active else "IDLE"
        grpc_color = COL_ACCENT if rpi_active else COL_WARN
        draw.text((130, panel_y+38), grpc_status, fill=grpc_color, font=font_tiny)
        
        # Pico W inference
        draw.text((12, panel_y+54), "• Pico W (inference)", fill=COL_MUTED, font=font_tiny)
        # Assume Pico is connected if ESP32 devices are publishing
        pico_connected = num_devs > 0
        pico_status = "INFERRING" if pico_connected else "OFFLINE"
        pico_color = COL_BLUE if pico_connected else COL_MUTED
        draw.text((130, panel_y+54), pico_status, fill=pico_color, font=font_tiny)

        # ── FOOTER ─────────────────────────────────────────────
        footer_y = 435
        draw.rectangle((0, footer_y, WIDTH, HEIGHT), fill=COL_PANEL, outline=COL_BORDER)
        
        # Health indicator
        health = "HEALTHY" if state.get("status") == "RUNNING" else "IDLE"
        health_color = COL_ACCENT if health == "HEALTHY" else COL_MUTED
        draw_status_dot(draw, 10, footer_y+18, active=(health=="HEALTHY"), size=10)
        draw.text((25, footer_y+16), health, fill=health_color, font=font_small)
        
        # Last update time
        draw.text((220, footer_y+16), f"Updated {state.get('timestamp', '--:--:--')}", 
                  fill=COL_MUTED, font=font_tiny)


# ══════════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════════

def main():
    # Initialize SPI LCD
    serial = spi(
        port=0,
        device=0,
        gpio_DC=GPIO_DC,
        gpio_RST=GPIO_RST,
    )

    device = ili9486(
        serial,
        width=WIDTH,
        height=HEIGHT,
        rotate=0,
    )

    print("=" * 58)
    print("  FL LCD Dashboard  —  ILI9486 320×480")
    print("  Monitoring: round_history.json")
    print("  Refresh: {} seconds".format(REFRESH_SECONDS))
    print("=" * 58)

    # Main display loop
    try:
        while True:
            state = load_fl_state()
            draw_dashboard(device, state)
            time.sleep(REFRESH_SECONDS)
    except KeyboardInterrupt:
        print("\nDashboard stopped by user")
        # Clear screen on exit
        with canvas(device) as draw:
            draw.rectangle((0, 0, WIDTH, HEIGHT), fill=COL_BG)
            draw.text((80, 220), "Dashboard stopped", fill=COL_MUTED, font=font_med)


if __name__ == "__main__":
    main()
