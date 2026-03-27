"""
strategy.py  —  runs on RPi #2  (same machine as edge_bridge.py)
=================================================================
Flower gRPC server + EdgeAwareFedAvg strategy.

Heterogeneous federation model
-------------------------------
This system runs two parallel model types in one federation:

  Model A  —  ESP32 linear regression  (7 weights)
              Managed by edge_bridge.py via MQTT
              ESP32s train locally, bridge averages their weights
              Global ESP32 weights pushed back after each round

  Model B  —  RPi TensorFlow model  (variable shape)
              Participates as a standard Flower NumPyClient
              FedAvg runs normally across RPi clients
              RPi #1 connects here via gRPC

The two models are NOT weight-averaged together — they are
fundamentally different architectures. Instead, metrics from
both are recorded per round so convergence of the heterogeneous
federation can be analyzed and presented.

This is the "shared metrics, separate models" pattern for
heterogeneous FL, which is a genuinely advanced FL topic.

Run on RPi #2:
    python strategy.py --rounds 20 --port 8080 --min-clients 1

RPi #1 connects as a standard Flower client:
    python rpi_client.py --server <rpi2_ip>:8080
"""

import argparse
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import flwr as fl
from flwr.common import (
    FitRes,
    EvaluateRes,
    Metrics,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from edge_bridge import (
    bridge,
    EDGE_BRIDGE_BUFFER,
    BUFFER_LOCK,
)

# ── logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-18s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("strategy")

# ── constants ─────────────────────────────────────────────────
METRICS_FILE      = "global_metrics.json"
ROUND_HISTORY_FILE = "round_history.json"


# ═══════════════════════════════════════════════════════════════
#  UTILITY
# ═══════════════════════════════════════════════════════════════

def pct(v: float, decimals: int = 5) -> float:
    """Convert 0–1 value to percentage with given decimal precision."""
    return round(float(v) * 100.0, decimals)

def _safe_avg(values: List[float]) -> Optional[float]:
    return float(sum(values) / len(values)) if values else None

def _save_json(path: str, data: Any) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ═══════════════════════════════════════════════════════════════
#  EDGE-AWARE FEDAVG STRATEGY
# ═══════════════════════════════════════════════════════════════

class EdgeAwareFedAvg(FedAvg):
    """
    FedAvg extended for heterogeneous federated learning.

    Each round does three things:
      1. Wait for ESP32 devices to submit their weights via
         edge_bridge (non-blocking from Flower's perspective —
         bridge handles the wait in a background thread before
         aggregate_fit is called)

      2. Run standard FedAvg across RPi TF clients (Model B)

      3. Read ESP32 metrics from EDGE_BRIDGE_BUFFER (Model A),
         log them alongside RPi metrics, and push the averaged
         ESP32 global weights back down to all devices

    The two model types converge independently. Per-round
    metrics for both are saved to round_history.json.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._round_history : List[Dict] = []
        self._last_esp32_weights: Optional[List[float]] = None

    # ── fit aggregation ───────────────────────────────────────

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        log.info(f"\n{'═'*58}")
        log.info(f"  ROUND {server_round}  —  aggregate_fit()")
        log.info(f"{'─'*58}")
        log.info(f"  RPi clients: {len(results)} responded, "
                 f"{len(failures)} failed")

        # ── Step 1: Wait for ESP32 round ──────────────────────
        # Bridge waits for all ESP32s to submit before we read.
        # This is a blocking call but runs on the Flower server
        # thread — Flower is already waiting in aggregate_fit.
        bridge.wait_for_esp32_round(server_round)

        # ── Step 2: FedAvg on RPi TF clients ──────────────────
        if failures:
            log.warning(f"  {len(failures)} RPi client(s) failed this round")

        # Standard FedAvg — returns (Parameters, metrics_dict)
        aggregated_params, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        # ── Step 3: Collect RPi client metrics ────────────────
        rpi_losses : List[float] = []
        rpi_accs   : List[float] = []

        for _, fit_res in results:
            m = fit_res.metrics or {}
            if "loss"     in m: rpi_losses.append(float(m["loss"]))
            if "accuracy" in m: rpi_accs.append(float(m["accuracy"]))

        rpi_avg_loss = _safe_avg(rpi_losses)
        rpi_avg_acc  = _safe_avg(rpi_accs)

        # ── Step 4: Read ESP32 metrics from bridge buffer ──────
        with BUFFER_LOCK:
            esp32_weights  = EDGE_BRIDGE_BUFFER.get("weights")
            esp32_metrics  = dict(EDGE_BRIDGE_BUFFER.get("metrics", {}))
            esp32_per_dev  = dict(EDGE_BRIDGE_BUFFER.get("per_device", {}))

        # ── Step 5: Push global weights back to ESP32s + Pico ──
        if esp32_weights is not None:
            self._last_esp32_weights = esp32_weights
            bridge.push_global_to_esp32s(esp32_weights)
        elif self._last_esp32_weights is not None:
            # No new ESP32 data this round — reuse last known weights
            log.warning("  No new ESP32 weights — pushing last known global")
            bridge.push_global_to_esp32s(self._last_esp32_weights)
        else:
            log.warning("  No ESP32 weights available yet — skipping push")

        # ── Step 6: Build round record ─────────────────────────
        round_record: Dict[str, Any] = {
            "round"    : server_round,
            "timestamp": time.time(),

            # Model B — RPi TF clients
            "rpi": {
                "num_clients"   : len(results),
                "num_failures"  : len(failures),
                "avg_loss"      : rpi_avg_loss,
                "avg_loss_pct"  : pct(rpi_avg_loss) if rpi_avg_loss else None,
                "avg_accuracy"  : rpi_avg_acc,
                "avg_acc_pct"   : pct(rpi_avg_acc)  if rpi_avg_acc  else None,
            },

            # Model A — ESP32 linear regression
            "esp32": {
                "num_devices" : int(esp32_metrics.get("num_devices", 0)),
                "avg_mse"     : esp32_metrics.get("loss_mse"),
                "avg_rmse"    : esp32_metrics.get("loss_rmse"),
                "avg_mae"     : esp32_metrics.get("loss_mae"),
                "avg_mse_pct" : pct(esp32_metrics["loss_mse"])
                                if "loss_mse" in esp32_metrics else None,
                "per_device"  : {
                    dev: {
                        "round"   : snap.get("round_num"),
                        "loss_mse": snap["metrics"].get("loss_mse"),
                        "loss_mae": snap["metrics"].get("loss_mae"),
                    }
                    for dev, snap in esp32_per_dev.items()
                },
            },
        }

        self._round_history.append(round_record)
        _save_json(ROUND_HISTORY_FILE, self._round_history)

        # ── Step 7: Console summary ────────────────────────────
        log.info(f"\n  ── Model B  (RPi TF) ──────────────────────────")
        if rpi_avg_loss is not None:
            log.info(f"     avg loss     : {pct(rpi_avg_loss):.5f}%")
        if rpi_avg_acc is not None:
            log.info(f"     avg accuracy : {pct(rpi_avg_acc):.5f}%")

        log.info(f"\n  ── Model A  (ESP32 linear) ────────────────────")
        if "loss_mse" in esp32_metrics:
            log.info(f"     avg MSE  : {esp32_metrics['loss_mse']:.6f}")
            log.info(f"     avg RMSE : {esp32_metrics.get('loss_rmse', 0):.6f}")
            log.info(f"     avg MAE  : {esp32_metrics.get('loss_mae',  0):.6f}")
            log.info(f"     devices  : {int(esp32_metrics.get('num_devices',0))}")
        else:
            log.info("     no ESP32 data this round")

        log.info(f"{'═'*58}\n")

        # Save latest combined metrics snapshot for quick access
        _save_json(METRICS_FILE, round_record)

        return aggregated_params, aggregated_metrics

    # ── evaluate aggregation ──────────────────────────────────

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:

        if not results:
            return None, {}

        losses = [res.loss for _, res in results]
        avg_loss = float(sum(losses) / len(losses))

        metrics: Dict[str, Scalar] = {
            "avg_loss"    : avg_loss,
            "avg_loss_pct": pct(avg_loss),
        }

        log.info(f"  Round {server_round} eval  "
                 f"avg_loss={pct(avg_loss):.5f}%  "
                 f"clients={len(results)}")

        # Annotate matching round record with eval data
        for record in reversed(self._round_history):
            if record["round"] == server_round:
                record["rpi"]["eval_loss"]     = avg_loss
                record["rpi"]["eval_loss_pct"] = pct(avg_loss)
                _save_json(ROUND_HISTORY_FILE, self._round_history)
                break

        return avg_loss, metrics

    # ── post-training summary ─────────────────────────────────

    def print_summary(self) -> None:
        if not self._round_history:
            return

        log.info("\n" + "═"*58)
        log.info("  FEDERATION COMPLETE — TRAINING SUMMARY")
        log.info("─"*58)
        log.info(f"  {'RND':>3}  {'RPi Loss%':>10}  "
                 f"{'RPi Acc%':>9}  {'ESP32 MSE':>10}  "
                 f"{'ESP32 Devs':>11}")
        log.info("  " + "─"*52)

        for r in self._round_history:
            rpi   = r.get("rpi",   {})
            esp32 = r.get("esp32", {})

            rpi_l  = f"{rpi['avg_loss_pct']:>10.5f}" \
                     if rpi.get("avg_loss_pct") is not None else "       n/a"
            rpi_a  = f"{rpi['avg_acc_pct']:>9.5f}" \
                     if rpi.get("avg_acc_pct")  is not None else "      n/a"
            esp_m  = f"{esp32['avg_mse']:>10.6f}" \
                     if esp32.get("avg_mse")     is not None else "       n/a"
            esp_d  = f"{esp32['num_devices']:>11}"

            log.info(f"  {r['round']:>3}  {rpi_l}  {rpi_a}  {esp_m}  {esp_d}")

        log.info("═"*58)
        log.info(f"  Round history saved → {ROUND_HISTORY_FILE}")
        log.info("═"*58 + "\n")


# ═══════════════════════════════════════════════════════════════
#  ENTRYPOINT
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="FL Heterogeneous Server — RPi #2"
    )
    parser.add_argument(
        "--rounds", type=int, default=20,
        help="Number of FL rounds  (default: 20)"
    )
    parser.add_argument(
        "--port", type=int, default=8080,
        help="gRPC listen port  (default: 8080)"
    )
    parser.add_argument(
        "--min-clients", type=int, default=1,
        help="Min RPi clients needed per round  (default: 1)"
    )
    args = parser.parse_args()

    # Start MQTT bridge daemon before Flower server opens
    log.info("Starting edge bridge (MQTT daemon)...")
    bridge.start()

    # Give ESP32s time to boot and connect to MQTT broker
    log.info("Waiting 5s for ESP32 devices to connect...")
    time.sleep(5)

    strategy = EdgeAwareFedAvg(
        min_fit_clients     = args.min_clients,
        min_evaluate_clients= args.min_clients,
        min_available_clients= args.min_clients,

        # No initial parameters — RPi client provides them on
        # first get_parameters() call
        initial_parameters=None,
    )

    address = f"0.0.0.0:{args.port}"
    log.info(f"Flower server starting on {address}")
    log.info(f"Rounds={args.rounds}  min_rpi_clients={args.min_clients}")
    log.info("ESP32 clients managed by edge_bridge (MQTT)")
    log.info("─"*58)

    fl.server.start_server(
        server_address=address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )

    strategy.print_summary()


if __name__ == "__main__":
    main()
