"""
edge_bridge.py  —  runs on RPi #2  (same machine as strategy.py)
=================================================================
MQTT daemon that manages all embedded device communication.
Imported and started by strategy.py before the Flower server
begins. Runs entirely in background daemon threads.

Responsibilities
----------------
  1. Connect to Mosquitto broker (localhost) and subscribe to
     ESP32 weight upload topics

  2. Maintain a per-device buffer of the latest weights and
     metrics from each ESP32

  3. Expose a shared EDGE_BRIDGE_BUFFER that strategy.py reads
     during aggregate_fit() to access ESP32 metrics

  4. After each Flower round completes, strategy.py calls
     push_global_to_esp32() — this sends the averaged ESP32
     global weights back down to all ESP32 devices so they
     warm-start the next round

  5. Push the ESP32 global model to the Pico W for inference
     after each round

Round synchronization
---------------------
  The ESP32s train asynchronously. edge_bridge controls timing:
    - ESP32 publishes weights  →  buffer is updated
    - strategy.aggregate_fit() reads buffer  →  injects metrics
    - strategy calls push_global_to_esp32()  →  ESP32s get new weights
    - ESP32s receive weights, train next round, publish again

  The bridge does NOT block Flower rounds. Flower rounds run
  on their own schedule (gRPC). The bridge simply ensures the
  most recent ESP32 weights are always available in the buffer.

MQTT topic map
--------------
  ESP32 → RPi #2   fl/esp32/{device_id}/update   trained weights
  RPi #2 → ESP32   fl/esp32/{device_id}/global   global weights back
  RPi #2 → Pico    fl/pico/global                global model for inference
"""

import json
import logging
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np
import paho.mqtt.client as mqtt

# ── logging ───────────────────────────────────────────────────
log = logging.getLogger("edge_bridge")

# ── MQTT config ───────────────────────────────────────────────
MQTT_BROKER    = "192.168.241.142"    # Mosquitto on same RPi #2
MQTT_PORT      = 1883
MQTT_KEEPALIVE = 60

# ── Device registry ───────────────────────────────────────────
# Must match DEVICE_ID defined in each ESP32 .ino sketch
DEVICE_IDS = ["esp32_01", "esp32_02"]

# Pico W inference topic
PICO_GLOBAL_TOPIC = "fl/pico/global"

# ── Model constants ───────────────────────────────────────────
ESP32_NUM_WEIGHTS = 7   # must match ESP32 sketch

# ── Round timeout ─────────────────────────────────────────────
# How long to wait for an ESP32 to publish weights before
# proceeding with whatever is already in the buffer.
# Should be longer than STEPS_PER_ROUND * loop_delay on ESP32
# (500 steps × 10ms = 5s, so 30s gives plenty of margin)
ESP32_ROUND_TIMEOUT = 30.0


# ═══════════════════════════════════════════════════════════════
#  TOPIC HELPERS
# ═══════════════════════════════════════════════════════════════

def _upload_topic(device_id: str) -> str:
    return f"fl/esp32/{device_id}/update"

def _download_topic(device_id: str) -> str:
    return f"fl/esp32/{device_id}/global"


# ═══════════════════════════════════════════════════════════════
#  PER-DEVICE BUFFER
#  One per ESP32. Holds latest weights + metrics.
#  Uses threading.Event for efficient blocking wait.
# ═══════════════════════════════════════════════════════════════

class _DeviceBuffer:

    def __init__(self, device_id: str):
        self.device_id  = device_id
        self._lock      = threading.Lock()
        self._event     = threading.Event()

        self.weights    : Optional[List[float]] = None
        self.metrics    : Dict[str, Any]        = {}
        self.timestamp  : Optional[float]       = None
        self.round_num  : int                   = 0

    def update(self, weights: List[float],
               metrics: Dict[str, Any], round_num: int) -> None:
        with self._lock:
            self.weights   = list(weights)
            self.metrics   = dict(metrics)
            self.timestamp = time.time()
            self.round_num = round_num
        self._event.set()

    def wait_for_update(self, timeout: float = ESP32_ROUND_TIMEOUT) -> bool:
        """
        Block until this ESP32 publishes new weights or timeout.
        Called by wait_for_all_esp32s() during round synchronization.
        """
        arrived = self._event.wait(timeout=timeout)
        if arrived:
            self._event.clear()
        return arrived

    def get_snapshot(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            if self.weights is None:
                return None
            return {
                "weights"  : list(self.weights),
                "metrics"  : dict(self.metrics),
                "timestamp": self.timestamp,
                "round_num": self.round_num,
            }

    @property
    def is_stale(self) -> bool:
        if self.timestamp is None:
            return True
        return (time.time() - self.timestamp) > 120.0

    @property
    def has_data(self) -> bool:
        return self.weights is not None


# ═══════════════════════════════════════════════════════════════
#  SHARED BUFFER  —  imported by strategy.py
#
#  strategy.py reads this dict in aggregate_fit() to access
#  the latest ESP32 weights and metrics without needing to
#  know anything about MQTT or device management.
#
#  Fields:
#    weights     averaged ESP32 weights (List[float] or None)
#    metrics     aggregated ESP32 metrics dict
#    round       last FL round this was updated for
#    per_device  per-device snapshots keyed by device_id
# ═══════════════════════════════════════════════════════════════

EDGE_BRIDGE_BUFFER: Dict[str, Any] = {
    "weights"   : None,
    "metrics"   : {},
    "round"     : 0,
    "per_device": {},
}

# Lock for strategy.py to use when reading EDGE_BRIDGE_BUFFER
BUFFER_LOCK = threading.Lock()


# ═══════════════════════════════════════════════════════════════
#  MQTT BRIDGE (internal — not imported directly by strategy.py)
# ═══════════════════════════════════════════════════════════════

class _MQTTBridge:

    def __init__(self):
        self._buffers: Dict[str, _DeviceBuffer] = {
            dev: _DeviceBuffer(dev) for dev in DEVICE_IDS
        }
        self._client  : Optional[mqtt.Client] = None
        self._started : bool = False
        self._lock    = threading.Lock()

    # ── lifecycle ─────────────────────────────────────────────

    def start(self) -> None:
        """
        Start the MQTT loop in a daemon thread.
        Called once from strategy.py at server startup.
        Subsequent calls are no-ops.
        """
        with self._lock:
            if self._started:
                return
            self._client = mqtt.Client(client_id="fl_edge_bridge_server")
            self._client.on_connect    = self._on_connect
            self._client.on_message    = self._on_message
            self._client.on_disconnect = self._on_disconnect
            self._client.connect(MQTT_BROKER, MQTT_PORT, MQTT_KEEPALIVE)
            t = threading.Thread(
                target=self._client.loop_forever,
                name="edge-bridge-mqtt",
                daemon=True,
            )
            t.start()
            self._started = True
            log.info(f"[edge_bridge] MQTT daemon started → "
                     f"{MQTT_BROKER}:{MQTT_PORT}")

    # ── public API (called by strategy.py) ───────────────────

    def wait_for_esp32_round(self, fl_round: int) -> None:
        """
        Wait for all ESP32 devices to submit weights for this round.
        Called by strategy.aggregate_fit() before reading the buffer.

        Devices that time out are skipped — their last known weights
        remain in the buffer from the previous round.
        """
        log.info(f"[edge_bridge] Waiting for ESP32 devices "
                 f"(round {fl_round}, timeout={ESP32_ROUND_TIMEOUT}s each)...")

        threads = []
        results = {}

        def _wait_one(dev_id: str) -> None:
            arrived = self._buffers[dev_id].wait_for_update(
                timeout=ESP32_ROUND_TIMEOUT
            )
            results[dev_id] = arrived
            if arrived:
                log.info(f"[edge_bridge] [{dev_id}] weights received ✓")
            else:
                log.warning(f"[edge_bridge] [{dev_id}] timed out — "
                            f"using previous round weights")

        for dev_id in DEVICE_IDS:
            t = threading.Thread(target=_wait_one, args=(dev_id,),
                                 daemon=True)
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        # Update the shared buffer with averaged ESP32 weights
        self._update_shared_buffer(fl_round)

    def push_global_to_esp32s(self, global_weights: List[float]) -> None:
        """
        Push averaged global ESP32 weights back to all devices.
        Called by strategy.py after aggregate_fit() completes.
        Also pushes the model to the Pico W for inference.
        """
        if not self._is_connected():
            log.error("[edge_bridge] Cannot push global weights — "
                      "MQTT not connected")
            return

        for dev_id in DEVICE_IDS:
            payload = json.dumps({
                "device_id": dev_id,
                "weights"  : global_weights,
                "timestamp": time.time(),
            })
            result = self._client.publish(
                _download_topic(dev_id), payload, qos=1
            )
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                log.info(f"[edge_bridge] [{dev_id}] global weights → "
                         f"{_download_topic(dev_id)}")
            else:
                log.error(f"[edge_bridge] [{dev_id}] publish failed "
                          f"rc={result.rc}")

        # Push same weights to Pico W for inference
        pico_payload = json.dumps({
            "weights"  : global_weights,
            "timestamp": time.time(),
        })
        result = self._client.publish(PICO_GLOBAL_TOPIC, pico_payload, qos=1)
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            log.info(f"[edge_bridge] Pico W global weights → "
                     f"{PICO_GLOBAL_TOPIC}")

    def get_device_snapshot(self, device_id: str) -> Optional[Dict]:
        return self._buffers[device_id].get_snapshot()

    # ── internal ──────────────────────────────────────────────

    def _update_shared_buffer(self, fl_round: int) -> None:
        """
        Average weights across all ESP32 devices that have data
        and write the result to EDGE_BRIDGE_BUFFER for strategy.py.

        Devices with no data or stale data are excluded from the
        average but their last known metrics are still recorded.
        """
        valid_weights = []
        valid_metrics = []
        per_device    = {}

        for dev_id, buf in self._buffers.items():
            snap = buf.get_snapshot()
            if snap is None:
                continue
            per_device[dev_id] = snap
            if not buf.is_stale:
                valid_weights.append(np.array(snap["weights"],
                                               dtype=np.float32))
                valid_metrics.append(snap["metrics"])

        if not valid_weights:
            log.warning("[edge_bridge] No valid ESP32 weights for buffer update")
            return

        # Average weights across devices (FedAvg across ESP32s)
        avg_weights = np.mean(valid_weights, axis=0).tolist()

        # Average numeric metrics
        agg_metrics: Dict[str, float] = {}
        for key in ["loss_mse", "loss_rmse", "loss_mae"]:
            vals = [m[key] for m in valid_metrics if key in m]
            if vals:
                agg_metrics[key] = float(sum(vals) / len(vals))
        agg_metrics["num_devices"] = float(len(valid_weights))

        with BUFFER_LOCK:
            EDGE_BRIDGE_BUFFER["weights"]    = avg_weights
            EDGE_BRIDGE_BUFFER["metrics"]    = agg_metrics
            EDGE_BRIDGE_BUFFER["round"]      = fl_round
            EDGE_BRIDGE_BUFFER["per_device"] = per_device

        log.info(f"[edge_bridge] Shared buffer updated  "
                 f"round={fl_round}  "
                 f"devices={len(valid_weights)}  "
                 f"mse={agg_metrics.get('loss_mse', 0.0):.6f}")

    # ── MQTT callbacks ────────────────────────────────────────

    def _on_connect(self, client, userdata, flags, rc: int) -> None:
        if rc != 0:
            log.error(f"[edge_bridge] MQTT connect failed rc={rc}")
            return
        log.info(f"[edge_bridge] MQTT connected to broker")
        for dev_id in DEVICE_IDS:
            topic = _upload_topic(dev_id)
            client.subscribe(topic, qos=1)
            log.info(f"[edge_bridge] subscribed → {topic}")

    def _on_message(self, client, userdata, msg) -> None:
        """
        Route incoming ESP32 weight payloads to the correct buffer.

        Expected payload fields (matching esp32_fl_light_v2.ino):
          device_id     str
          round         int
          weights       List[float]  length == ESP32_NUM_WEIGHTS
          loss_mse      float
          loss_rmse     float
          loss_mae      float
          samples       int
          weight_deltas List[float]  (optional)
        """
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            log.error(f"[edge_bridge] Bad MQTT payload on {msg.topic}: {e}")
            return

        device_id = payload.get("device_id")
        if device_id not in self._buffers:
            log.warning(f"[edge_bridge] Unknown device '{device_id}' — "
                        f"add to DEVICE_IDS to register it")
            return

        weights = payload.get("weights")
        if not isinstance(weights, list) or len(weights) != ESP32_NUM_WEIGHTS:
            log.error(f"[edge_bridge] [{device_id}] invalid weights "
                      f"(expected {ESP32_NUM_WEIGHTS}, "
                      f"got {len(weights) if isinstance(weights, list) else '?'})")
            return

        metrics = {
            "loss_mse"     : float(payload.get("loss_mse",  0.0)),
            "loss_rmse"    : float(payload.get("loss_rmse", 0.0)),
            "loss_mae"     : float(payload.get("loss_mae",  0.0)),
            "samples"      : int(payload.get("samples",     0)),
            "weight_deltas": payload.get("weight_deltas",   []),
        }

        self._buffers[device_id].update(
            weights, metrics, payload.get("round", 0)
        )
        log.info(f"[edge_bridge] [{device_id}] "
                 f"round={payload.get('round', 0)}  "
                 f"mse={metrics['loss_mse']:.6f}  "
                 f"samples={metrics['samples']}")

    def _on_disconnect(self, client, userdata, rc: int) -> None:
        if rc != 0:
            log.warning(f"[edge_bridge] Unexpected MQTT disconnect "
                        f"rc={rc} — paho will auto-reconnect")

    def _is_connected(self) -> bool:
        return (self._client is not None
                and self._client.is_connected())


# ── Module-level singleton ─────────────────────────────────────
# strategy.py calls:  bridge.start()
#                     bridge.wait_for_esp32_round(server_round)
#                     bridge.push_global_to_esp32s(weights)
bridge = _MQTTBridge()
