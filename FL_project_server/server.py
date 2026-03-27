import logging
import time
import flwr as fl

from edge_bridge import bridge
from strategy import EdgeAwareFedAvg

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-18s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("server")

if __name__ == "__main__":
    # Start MQTT edge bridge daemon
    log.info("Starting edge bridge (MQTT daemon)...")
    bridge.start()
    
    # Wait for ESP32 devices to boot and connect
    log.info("Waiting 5s for ESP32 devices to connect to MQTT...")
    time.sleep(5)

    # Create strategy
    strategy = EdgeAwareFedAvg(
        fraction_fit=1.0,           # use all available clients
        min_fit_clients=1,          # need at least 1 RPi TF client
        min_available_clients=1,    # start rounds with 1 client
        min_evaluate_clients=1,
        initial_parameters=None,    # RPi client provides on first call
    )

    # Start Flower gRPC server
    server_address = "0.0.0.0:8080"
    num_rounds     = 100

    log.info(f"Flower server starting on {server_address}")
    log.info(f"Rounds={num_rounds}  min_rpi_clients=1")
    log.info("ESP32 clients managed by edge_bridge via MQTT")
    log.info("-" * 58)

    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    # Print summary after training completes
    strategy.print_summary()
    
    log.info("Server shutdown complete")
