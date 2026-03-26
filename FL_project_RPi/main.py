import network
import time
import ujson
from simple import MQTTClient
from machine import Pin
import utime
from machine import Pin, SoftI2C
import ssd1306
import machine
import math

i2c = SoftI2C(scl=Pin(5), sda=Pin(4))

oled_width = 128
oled_height = 64
oled = ssd1306.SSD1306_I2C(oled_width, oled_height, i2c)

# =========================
# CONFIG
# =========================
WIFI_SSID = "THE STAR ARTS"
WIFI_PASS = "NISSANFRONTIER1"
led = machine.Pin("LED", machine.Pin.OUT)
BROKER = "192.168.241.142"  # RPi #2 server IP - change if needed
TOPIC = "fl/pico/global"
CLIENT_ID = "pico_infer"

# Model constants - must match ESP32
NUM_FEATURES = 6
NUM_WEIGHTS = 7  # 6 features + 1 bias

# =========================
# GLOBALS
# =========================
weights = []
ip_address = "Not connected"
model_status = "Initializing..."
inference_count = 0
last_prediction = None
mqtt_connected = False
last_model_update = None
sim_hour = 0.0  # Simulated hour of day (0-24)

def blink():
    global led
    for _ in range(3):
        led.value(1)
        utime.sleep(0.3)
        led.value(0)
        utime.sleep(0.3)

# =========================
# OLED DISPLAY HELPER
# =========================
def display_status():
    """Update OLED with current status information"""
    try:
        oled.fill(0)
        
        # Line 1: IP Address (always show)
        ip_short = ip_address if len(ip_address) <= 15 else ip_address.split('.')[-1]
        oled.text(f"IP:{ip_short}", 0, 0)
        
        # Line 2: WiFi & MQTT Status
        w = "W:OK" if ip_address != "Not connected" else "W:NO"
        m = "M:Y" if mqtt_connected else "M:N"
        oled.text(f"{w} {m}", 0, 8)
        
        # Line 3: Model Status
        oled.text(f"M:{model_status[:12]}", 0, 16)
        
        # Line 4: Weights info (if available)
        if len(weights) == NUM_WEIGHTS:
            oled.text(f"W:{NUM_WEIGHTS} OK", 0, 24)
        else:
            oled.text(f"W:Wait", 0, 24)
        
        # Line 5: Inference count
        oled.text(f"Inf:{inference_count}", 0, 32)
        
        # Line 6: Current simulated hour
        oled.text(f"H:{sim_hour:.1f}h", 0, 40)
        
        # Line 7: Last prediction (if available)
        if last_prediction is not None:
            pred_pct = int(last_prediction * 100)
            oled.text(f"L:{pred_pct}%", 0, 48)
            # Simple bar (0-60 pixels)
            bar_len = int(last_prediction * 60)
            if bar_len > 60:
                bar_len = 60
            for x in range(bar_len):
                oled.pixel(65 + x, 50, 1)
        else:
            oled.text("L:Wait", 0, 48)
        
        # Line 8: LED indicator
        led_txt = "LED:ON" if led.value() else "LED:OFF"
        oled.text(led_txt, 0, 56)
        
        oled.show()
    except Exception as e:
        oled.fill(0)
        oled.text("ERR", 0, 0)
        oled.text(str(e)[:12], 0, 8)
        oled.show()

def display_error(title, error_msg=""):
    """Display error message on OLED"""
    try:
        oled.fill(0)
        oled.text("ERR", 0, 0)
        oled.text(title[:12], 0, 8)
        if error_msg:
            oled.text(str(error_msg)[:12], 0, 16)
        oled.text("Reconnect", 0, 24)
        oled.show()
    except:
        pass

# =========================
# WIFI
# =========================
def connect_wifi(max_attempts=50):
    global ip_address, model_status
    
    model_status = "WiFi"
    oled.fill(0)
    oled.text("Boot", 0, 20)
    oled.text("FL Inference", 0, 30)
    oled.show()
    time.sleep(1)
    
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    
    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        oled.fill(0)
        oled.text("WiFi...", 0, 20)
        oled.text(f"Try:{attempt}/50", 0, 30)
        oled.show()
        
        if not wlan.isconnected():
            wlan.connect(WIFI_SSID, WIFI_PASS)
        
        # Wait up to 5 seconds per attempt
        for _ in range(10):
            if wlan.isconnected():
                # Get IP address
                ip_address = wlan.ifconfig()[0]
                model_status = "Ready"
                
                oled.fill(0)
                oled.text("WiFi OK", 0, 10)
                oled.text(f"IP:{ip_address}", 0, 20)
                oled.text("MQTT...", 0, 40)
                oled.show()
                blink()
                return True
            time.sleep(0.5)
    
    # Failed after max attempts
    model_status = "WiFi Fail"
    display_error("WiFi", "50 Fail")
    return False


# =========================
# FEATURE EXTRACTION
# Must match ESP32's build_features()
# =========================
def build_features(hour):
    """
    Build 6-element feature vector from hour of day.
    Exactly matches ESP32 linear model feature extraction.
    
    Returns: list of 6 floats
    """
    two_pi_h = 2.0 * math.pi * hour / 24.0
    four_pi_h = 4.0 * math.pi * hour / 24.0
    pi_h = math.pi * hour / 24.0
    
    return [
        math.sin(two_pi_h),         # x[0] - fundamental sin
        math.cos(two_pi_h),         # x[1] - fundamental cos
        math.sin(four_pi_h),        # x[2] - 2nd harmonic sin
        math.cos(four_pi_h),        # x[3] - 2nd harmonic cos
        math.sin(pi_h) ** 2,        # x[4] - daylight arc
        hour / 24.0,                # x[5] - linear trend
    ]


# =========================
# INFERENCE
# =========================
def predict(hour):
    """
    Run inference on synthetic time-of-day input.
    
    Args:
        hour: float (0-24) - simulated hour of day
    
    Returns:
        float: predicted light level [0, 1] or None if no model
    """
    global inference_count, last_prediction
    
    if len(weights) != NUM_WEIGHTS:
        last_prediction = None
        return None
    
    # Build feature vector
    features = build_features(hour)
    
    # Linear model: y = w[0]*x[0] + w[1]*x[1] + ... + w[5]*x[5] + w[6]
    result = weights[NUM_WEIGHTS - 1]  # Start with bias (w[6])
    for i in range(NUM_FEATURES):
        result += weights[i] * features[i]
    
    # Clamp to [0, 1] range
    if result < 0.0:
        result = 0.0
    if result > 1.0:
        result = 1.0
    
    inference_count += 1
    last_prediction = result
    return result


# =========================
# MQTT CALLBACK
# =========================
def on_message(topic, msg):
    global weights, model_status, last_model_update
    try:
        data = ujson.loads(msg)
        new_weights = data.get("weights", [])
        
        # Validate weight count
        if len(new_weights) != NUM_WEIGHTS:
            model_status = f"Bad:{len(new_weights)}W"
            oled.fill(0)
            oled.text("Model Error", 0, 10)
            oled.text(f"Got:{len(new_weights)}W", 0, 20)
            oled.text(f"Need:{NUM_WEIGHTS}W", 0, 30)
            oled.show()
            time.sleep(2)
            return
        
        weights = new_weights
        model_status = "Updated"
        last_model_update = time.time()
        
        # Show update notification
        oled.fill(0)
        oled.text("Model Update", 0, 10)
        oled.text(f"W:{NUM_WEIGHTS} OK", 0, 20)
        oled.text(f"W0:{weights[0]:.2f}", 0, 30)
        oled.text(f"W6:{weights[6]:.2f}", 0, 40)
        oled.show()
        blink()
        time.sleep(2)
        
        model_status = "Run"
        
    except ValueError as e:
        model_status = "JSON Err"
        display_error("JSON", str(e)[:8])
        time.sleep(2)
    except Exception as e:
        model_status = "Parse Err"
        display_error("Upd", str(e)[:8])
        time.sleep(2)


# =========================
# MAIN
# =========================
if not connect_wifi():
    display_error("WiFi", "Fail")
    while True:
        time.sleep(5)

try:
    
    mqtt_attempt = 0
    max_mqtt_attempts = 50
    client = None
    
    while mqtt_attempt < max_mqtt_attempts:
        mqtt_attempt += 1
        try:
            model_status = "MQTT"
            oled.fill(0)
            oled.text("MQTT...", 0, 20)
            oled.text(f"Try:{mqtt_attempt}/50", 0, 30)
            oled.show()
            time.sleep(0.5)
            
            client = MQTTClient(CLIENT_ID, BROKER)
            client.set_callback(on_message)
            client.connect()
            client.subscribe(TOPIC)
            
            mqtt_connected = True
            model_status = "OK"
            
            blink()
            
            oled.fill(0)
            oled.text("MQTT OK", 0, 10)
            oled.text(CLIENT_ID[:14], 0, 20)
            oled.text("Wait Model", 0, 30)
            oled.text("from server", 0, 40)
            oled.show()
            time.sleep(2)
            
            model_status = "Wait"
            break
            
        except Exception as e:
            mqtt_connected = False
            time.sleep(1)
            continue
    
    if mqtt_attempt >= max_mqtt_attempts or client is None:
        mqtt_connected = False
        model_status = "Fail"
        display_error("MQTT", "50 Fail")
        while True:
            time.sleep(5)

except Exception as e:
    mqtt_connected = False
    model_status = "Fail"
    display_error("MQTT", str(e)[:8])
    while True:
        time.sleep(5)

# ============================
# MAIN INFERENCE LOOP
# ============================

# Initialize simulated hour to a random starting point
# so multiple Picos don't show identical predictions
try:
    import urandom
    # Random hour between 0-24
    sim_hour = float(urandom.getrandbits(11) % 1440) / 60.0
except:
    # Fallback if urandom not available
    sim_hour = 8.0

last_display_update = time.time()
last_inference = time.time()
display_update_interval = 1.0   # Update display every 1 second
inference_interval = 2.0        # Run inference every 2 seconds

print(f"Starting inference loop at sim_hour={sim_hour:.2f}")

while True:
    try:
        # Check for new model updates from MQTT
        client.check_msg()
        
        current_time = time.time()
        
        # Run inference periodically
        if current_time - last_inference >= inference_interval:
            # Advance simulated time by 2 minutes per inference cycle
            # (matches ESP32 training clock: 2 minutes per step)
            sim_hour = (sim_hour + 2.0 / 60.0) % 24.0
            
            # Run prediction if model is available
            if len(weights) == NUM_WEIGHTS:
                prediction = predict(sim_hour)
                
                if prediction is not None:
                    # Control LED based on prediction
                    # Threshold: light level > 0.5 = LED ON
                    if prediction > 0.5:
                        led.value(1)
                    else:
                        led.value(0)
            else:
                # No model yet - blink LED slowly to show alive
                led.value(1 if int(current_time) % 2 == 0 else 0)
            
            last_inference = current_time
        
        # Update display periodically
        if current_time - last_display_update >= display_update_interval:
            display_status()
            last_display_update = current_time
        
        time.sleep(0.1)  # Small sleep to prevent busy-wait
        
    except OSError as e:
        # Network error - attempt MQTT reconnect
        mqtt_connected = False
        model_status = "Lost"
        led.value(0)
        display_error("Lost", "Retry")
        time.sleep(3)
        
        # Attempt to reconnect MQTT
        retry_mqtt = 0
        while retry_mqtt < 50:
            retry_mqtt += 1
            try:
                oled.fill(0)
                oled.text("Reconnect", 0, 20)
                oled.text(f"Try:{retry_mqtt}/50", 0, 30)
                oled.show()
                
                client = MQTTClient(CLIENT_ID, BROKER)
                client.set_callback(on_message)
                client.connect()
                client.subscribe(TOPIC)
                
                mqtt_connected = True
                model_status = "Recon"
                
                oled.fill(0)
                oled.text("Reconnected", 0, 20)
                oled.text("Back online", 0, 30)
                oled.show()
                blink()
                time.sleep(1)
                break
                
            except:
                time.sleep(2)
                continue
        
        if retry_mqtt >= 50:
            display_error("Fail", "Reboot")
            time.sleep(5)
        
    except Exception as e:
        model_status = "ERR"
        led.value(0)
        display_error("Run", str(e)[:8])
        time.sleep(5)

