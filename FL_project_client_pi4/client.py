import random
import logging

import numpy as np
import tensorflow as tf
import flwr as fl

# ── logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [RPI-TF-CLIENT]  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("rpi_tf_client")

# Suppress TensorFlow info/warning spam
tf.get_logger().setLevel('ERROR')

# ══════════════════════════════════════════════════════════════
#  NETWORK CONFIG
# ══════════════════════════════════════════════════════════════

def get_server_address() -> str:
    """
    Prompt for server IP if running interactively, otherwise
    use environment variable or default to localhost.
    """
    try:
        import os
        server_ip = os.environ.get("FL_SERVER_IP")
        if not server_ip:
            server_ip = input("Enter FL server IP (default: localhost): ").strip()
        if not server_ip:
            server_ip = "localhost"
        return f"{server_ip}:8080"
    except (EOFError, KeyboardInterrupt):
        # Non-interactive environment or user cancelled
        log.info("Using default server address: localhost:8080")
        return "localhost:8080"


# ══════════════════════════════════════════════════════════════
#  DATA GENERATION
#  Simulates DHT11/DHT22 sensor readings (temp, humidity, light).
#  Label: 1 if dangerous conditions (high temp OR low humidity)
# ══════════════════════════════════════════════════════════════

def generate_data(samples: int = 200):
    """
    Generate synthetic environmental sensor data.
    
    Features:
      [0] temperature (°C)     15–40 range
      [1] humidity    (%)      20–90 range
      [2] light       (lux)    0–1000 range (normalised below)
    
    Label:
      1 if temp > 35 OR humidity < 25 (dangerous)
      0 otherwise
    
    Returns
    -------
    X : np.ndarray  shape (samples, 3)
    y : np.ndarray  shape (samples,)
    """
    X = np.zeros((samples, 3), dtype=np.float32)
    y = np.zeros(samples, dtype=np.float32)
    
    for i in range(samples):
        temp     = random.uniform(15.0, 40.0)
        humidity = random.uniform(20.0, 90.0)
        light    = random.uniform(0.0, 1000.0)
        
        # Normalise light to [0, 1] range to match other features
        light_norm = light / 1000.0
        
        # Danger label: high temp OR low humidity
        danger = 1.0 if (temp > 35.0 or humidity < 25.0) else 0.0
        
        X[i] = [temp, humidity, light_norm]
        y[i] = danger
    
    return X, y


# ══════════════════════════════════════════════════════════════
#  MODEL
# ══════════════════════════════════════════════════════════════

def create_model() -> tf.keras.Model:
    """
    Build a small feedforward neural network for binary
    classification of environmental danger.
    
    Architecture:
      Input(3) → Dense(16, relu) → Dense(8, relu) → Dense(1, sigmoid)
    
    This is Model B in the heterogeneous federation.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(3,)),
        tf.keras.layers.Dense(16, activation="relu",
                              kernel_initializer="he_normal"),
        tf.keras.layers.Dense(8,  activation="relu",
                              kernel_initializer="he_normal"),
        tf.keras.layers.Dense(1,  activation="sigmoid"),
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    
    return model


# ══════════════════════════════════════════════════════════════
#  BANDWIDTH ESTIMATION (optional metric)
# ══════════════════════════════════════════════════════════════

def estimate_bytes(weights) -> int:
    """Calculate total bytes in weight arrays."""
    return sum(w.nbytes for w in weights)


# ══════════════════════════════════════════════════════════════
#  FLOWER CLIENT
# ══════════════════════════════════════════════════════════════

class RPiTFClient(fl.client.NumPyClient):
    """
    Standard Flower NumPyClient for the RPi TensorFlow model.
    
    Each round:
      1. Receives global weights from server
      2. Generates fresh random training data (simulating new
         sensor readings arriving each round)
      3. Trains locally for 1 epoch
      4. Returns updated weights + metrics to server
    
    This mirrors the ESP32 pattern where each device trains on
    new local data every round rather than a fixed dataset.
    """
    
    def __init__(self):
        self.model = create_model()
        self._round = 0
        log.info("RPi TF client initialised")
        log.info(f"Model params: {self.model.count_params()}")
    
    def get_parameters(self, config):
        """Return current model weights."""
        return self.model.get_weights()
    
    def fit(self, parameters, config):
        """
        Train the model for one round.
        
        Parameters from server are the global model after FedAvg.
        Fresh synthetic data is generated each round to simulate
        continuous sensor readings.
        """
        self._round += 1
        log.info(f"-- Fit round {self._round} --")
        
        # Accept global model from server
        self.model.set_weights(parameters)
        
        # Generate fresh data for this round (200 samples)
        X_train, y_train = generate_data(samples=200)
        
        # Local training (1 epoch)
        history = self.model.fit(
            X_train, y_train,
            epochs=1,
            batch_size=32,
            verbose=0,
        )
        
        # Extract metrics
        loss     = float(history.history["loss"][0])
        accuracy = float(history.history["accuracy"][0])
        
        weights      = self.model.get_weights()
        bytes_sent   = estimate_bytes(weights)
        
        log.info(f"Fit complete  loss={loss:.6f}  "
                 f"acc={accuracy:.6f}  bytes={bytes_sent}")
        
        return weights, len(X_train), {
            "loss"      : loss,
            "accuracy"  : accuracy,
            "bytes_sent": bytes_sent,
        }
    
    def evaluate(self, parameters, config):
        """
        Evaluate the global model on fresh generated data.
        
        This is called by Flower after each round if the server
        strategy has evaluate enabled.
        """
        self.model.set_weights(parameters)
        
        # Fresh evaluation data (100 samples)
        X_eval, y_eval = generate_data(samples=100)
        
        loss, acc = self.model.evaluate(X_eval, y_eval, verbose=0)
        
        log.info(f"Eval  loss={loss:.6f}  acc={acc:.6f}")
        
        return float(loss), len(X_eval), {
            "loss"    : float(loss),
            "accuracy": float(acc),
        }


# ══════════════════════════════════════════════════════════════
#  ENTRYPOINT
# ══════════════════════════════════════════════════════════════

def main():
    log.info("="*58)
    log.info("  RPi TensorFlow Federated Learning Client")
    log.info("  Model: Binary classification (DHT danger detection)")
    log.info("="*58)
    
    server_address = get_server_address()
    log.info(f"Connecting to Flower server: {server_address}")
    
    try:
        fl.client.start_numpy_client(
            server_address=server_address,
            client=RPiTFClient(),
        )
    except KeyboardInterrupt:
        log.info("Client stopped by user")
    except Exception as e:
        log.error(f"Client error: {e}")
        raise


if __name__ == "__main__":
    main()
