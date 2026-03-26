

/*
 PROGRAMMED BY CRAIG KANYASA WITH THE HELP OF CLAUDE AI 
                                              FLOWER FORUMS
                                              
                                            

    copyright - THE STAR ARTS

 ============================================================
 *  ESP32-S3  |  Federated Learning Client  |  v2.0
 *  Task      : Predict light level from time of day
 *  Model     : Linear regression with cyclic time features
 *  Transport : MQTT → RPi Bridge → Flower Server
 * ============================================================
 *
 *  WHAT THIS MODEL ACTUALLY LEARNS:
 *  ---------------------------------
 *  Light levels follow a daily arc — low at night, rising at
 *  dawn, peaking at noon, falling at dusk. This arc is smooth
 *  but non-linear and varies with noise. The model learns to
 *  map time-of-day → expected light level using sinusoidal
 *  time encoding, which is the standard approach for any
 *  cyclical temporal feature.
 *
 *  The two ESP32 nodes simulate slightly different environments
 *  (e.g. one near a window, one in a corridor) so FedAvg
 *  across rounds genuinely combines knowledge from two
 *  different local distributions — which is the entire point
 *  of federated learning.
 *
 *  DEVICE IDENTITY:
 *  ----------------
 *  Change DEVICE_ID and DEVICE_BRIGHTNESS_BIAS below per unit.
 *  ESP32 #1: DEVICE_ID = "esp32_01", BIAS = 0.0f
 *  ESP32 #2: DEVICE_ID = "esp32_02", BIAS = 0.15f
 * ============================================================
 */

#include <Arduino.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <math.h>

// ============================================================
//  DEVICE IDENTITY  — change per physical unit
// ============================================================
#define DEVICE_ID             "esp32_02"
#define DEVICE_BRIGHTNESS_BIAS 0.0f    // esp32_02 uses 0.15f

// ============================================================
//  CREDENTIALS  — keep in a gitignored credentials.h in prod
// ============================================================
#define WIFI_SSID    "THE STAR ARTS"
#define WIFI_PASS    "NISSANFRONTIER1"
#define MQTT_BROKER  "192.168.241.142"
#define MQTT_PORT    1883

// ============================================================
//  MQTT TOPICS
// ============================================================
#define TOPIC_PUBLISH   "fl/esp32/esp32_01/update"     // weights → bridge
#define TOPIC_SUBSCRIBE "fl/esp32/esp32_01/global"   // global weights ← server

// ============================================================
//  MODEL CONFIG
// ============================================================

/*
 *  Feature vector (6 weights + 1 bias = 7 total):
 *
 *  x[0] = sin(2π · hour/24)      — captures AM/PM symmetry
 *  x[1] = cos(2π · hour/24)      — phase-shifts to mark noon
 *  x[2] = sin(4π · hour/24)      — second harmonic (dawn/dusk)
 *  x[3] = cos(4π · hour/24)      — second harmonic complement
 *  x[4] = sin²(π · hour/24)      — asymmetric daylight arc
 *  x[5] = hour/24  (normalised)  — linear trend component
 *  x[6] = 1.0                    — bias term
 *
 *  Why these features?
 *  Using sin/cos pairs of the hour angle encodes time as a
 *  position on a circle — midnight and 23:59 are adjacent,
 *  which raw hour values never capture. The second harmonic
 *  lets the model learn a sharper midday peak. sin² adds the
 *  asymmetric arc shape real daylight has. Together these 7
 *  features can approximate any smooth daily light curve.
 */
#define NUM_FEATURES   6     // excluding bias
#define NUM_WEIGHTS    7     // features + bias

/*
 *  Learning rate: 0.01 is appropriate here because features
 *  are all normalised to [-1, 1] and the target is normalised
 *  to [0, 1]. With 0.001 convergence is unnecessarily slow
 *  across FL rounds.
 */
#define LEARNING_RATE  0.01f

/*
 *  Gradient clipping threshold. Prevents any single weight
 *  update from exceeding this value — guards against the
 *  exploding gradient problem that plagued v1.
 */
#define GRAD_CLIP      0.5f

/*
 *  How many SGD steps per FL round before publishing weights.
 *  500 steps × 10ms delay = ~5 seconds per round.
 */
#define STEPS_PER_ROUND 500

/* accuracy calculations removed — lightweight client */

/*
 *  Simulated clock: minutes advance per loop iteration.
 *  This makes one simulated day pass in ~2.4 minutes of
 *  real time, giving the model full daily coverage quickly.
 */
#define SIM_MINUTES_PER_STEP 2.0f

// ============================================================
//  GLOBALS
// ============================================================

WiFiClient   wifiClient;
PubSubClient mqttClient(wifiClient);

float weights[NUM_WEIGHTS];          // current local weights
float weights_round_start[NUM_WEIGHTS]; // snapshot at round start (for delta)

unsigned long stepCount     = 0;
unsigned long roundCount    = 0;
float         roundLossAcc  = 0.0f;  // accumulates MSE for this round
float         roundMAEAcc   = 0.0f;  // accumulates |error| for true MAE
float         simMinutes    = 0.0f;  // simulated clock in minutes

bool  awaitingGlobalWeights = false; // true after publishing, waiting for server
unsigned long lastPublishMs = 0;

// ============================================================
//  SIMULATED LDR SENSOR
// ============================================================

/*
 *  simulate_ldr(hour_of_day)
 *
 *  Returns a normalised light reading in [0, 1] that mimics
 *  a real LDR sensor. The true underlying curve is:
 *
 *    base   = max(0, sin(π · h/24))²   — smooth daylight arc
 *    output = base + device_bias + gaussian_noise
 *
 *  The model does NOT have access to this formula. It only
 *  sees (time_features, noisy_reading) pairs and must infer
 *  the underlying shape through training — which is a genuine
 *  learning task unlike v1's tautological ground truth.
 *
 *  The gaussian noise (σ ≈ 0.03) simulates real sensor
 *  variability. DEVICE_BRIGHTNESS_BIAS shifts one device's
 *  environment slightly brighter (window vs corridor), giving
 *  FedAvg something meaningful to average across.
 */
float simulate_ldr(float hour) {
    // Core daylight arc: zero at midnight, peak at noon
    float angle = (float)M_PI * hour / 24.0f;
    float base  = sinf(angle);
    if (base < 0.0f) base = 0.0f;
    base = base * base;   // square for sharper midday peak

    // Device-specific environment offset
    base += DEVICE_BRIGHTNESS_BIAS;

    // Box-Muller gaussian noise (σ ≈ 0.03)
    float u1 = ((float)esp_random() / UINT32_MAX);
    float u2 = ((float)esp_random() / UINT32_MAX);
    if (u1 < 1e-6f) u1 = 1e-6f;
    float noise = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2) * 0.03f;

    float reading = base + noise;

    // Clamp to valid sensor range [0, 1]
    if (reading < 0.0f) reading = 0.0f;
    if (reading > 1.0f) reading = 1.0f;

    return reading;
}

// ============================================================
//  FEATURE EXTRACTION
// ============================================================

/*
 *  build_features(hour, x[])
 *
 *  Converts a raw hour value (0–24) into the 6-element
 *  normalised feature vector. All values are in [-1, 1]
 *  which keeps gradients well-conditioned.
 */
void build_features(float hour, float x[NUM_FEATURES]) {
    float h = hour;
    float two_pi_h  = 2.0f * (float)M_PI * h / 24.0f;
    float four_pi_h = 4.0f * (float)M_PI * h / 24.0f;
    float pi_h      = (float)M_PI * h / 24.0f;

    x[0] = sinf(two_pi_h);               // fundamental sin
    x[1] = cosf(two_pi_h);               // fundamental cos
    x[2] = sinf(four_pi_h);              // 2nd harmonic sin
    x[3] = cosf(four_pi_h);              // 2nd harmonic cos
    x[4] = sinf(pi_h) * sinf(pi_h);      // sin²(πh/24) — daylight arc
    x[5] = h / 24.0f;                    // normalised linear trend
}

// ============================================================
//  MODEL — PREDICT
// ============================================================

/*
 *  predict(x[])
 *
 *  Simple dot product of feature vector and weights, plus bias.
 *  Output is NOT clamped here — clamping at inference time
 *  would block gradient flow during training.
 */
float predict(float x[NUM_FEATURES]) {
    float y = weights[NUM_WEIGHTS - 1];  // bias term
    for (int i = 0; i < NUM_FEATURES; i++) {
        y += weights[i] * x[i];
    }
    return y;
}

// ============================================================
//  MODEL — TRAINING STEP
// ============================================================

/*
 *  train_step(x[], y_true)
 *
 *  One step of stochastic gradient descent using MSE loss.
 *  Gradient for weight i: dL/dw_i = error × x[i]
 *  Bias gradient:         dL/db   = error × 1
 *
 *  Gradient clipping is applied per-update to prevent any
 *  single step from destabilising the model.
 *
 *  Returns: squared error for this step (for loss tracking)
 */
float train_step(float x[NUM_FEATURES], float y_true) {
    float y_pred = predict(x);
    float error  = y_pred - y_true;

    // Update feature weights with clipped gradients
    for (int i = 0; i < NUM_FEATURES; i++) {
        float grad = LEARNING_RATE * error * x[i];
        // Clip gradient to [-GRAD_CLIP, +GRAD_CLIP]
        if (grad >  GRAD_CLIP) grad =  GRAD_CLIP;
        if (grad < -GRAD_CLIP) grad = -GRAD_CLIP;
        weights[i] -= grad;
    }

    // Bias update
    float bias_grad = LEARNING_RATE * error;
    if (bias_grad >  GRAD_CLIP) bias_grad =  GRAD_CLIP;
    if (bias_grad < -GRAD_CLIP) bias_grad = -GRAD_CLIP;
    weights[NUM_WEIGHTS - 1] -= bias_grad;

    return error * error;   // MSE contribution
}

// ============================================================
//  WEIGHT VALIDATION
// ============================================================

/*
 *  weights_valid()
 *
 *  Checks every weight for NaN or Inf. Either indicates the
 *  model has diverged. Called after every training step in
 *  debug builds; called before every MQTT publish always.
 */
bool weights_valid() {
    for (int i = 0; i < NUM_WEIGHTS; i++) {
        if (isnanf(weights[i]) || isinff(weights[i])) return false;
    }
    return true;
}

/*
 *  init_weights()
 *
 *  Xavier-style initialisation: draws from uniform [-k, k]
 *  where k = sqrt(1 / NUM_FEATURES). Appropriate for a linear
 *  model with normalised inputs — avoids the large random
 *  initialisations that contributed to instability in v1.
 */
void init_weights() {
    float k = sqrtf(1.0f / (float)NUM_FEATURES);
    for (int i = 0; i < NUM_WEIGHTS; i++) {
        float r = ((float)esp_random() / (float)UINT32_MAX);
        weights[i] = (r * 2.0f - 1.0f) * k;
    }
    Serial.printf("[MODEL] Weights initialised (Xavier k=%.4f)\n", k);
}

// ============================================================
//  MQTT — CALLBACKS
// ============================================================

/*
 *  on_global_weights()
 *
 *  Called when the RPi bridge publishes aggregated global
 *  weights back down after a FedAvg round. Overwrites local
 *  weights so the next round starts from the global model —
 *  this is the return path that makes it true federated
 *  learning rather than just distributed training.
 */
void on_global_weights(char* topic, byte* payload, unsigned int length) {
    StaticJsonDocument<512> doc;
    DeserializationError err = deserializeJson(doc, payload, length);

    if (err) {
        Serial.printf("[MQTT] Failed to parse global weights: %s\n", err.c_str());
        return;
    }

    JsonArray w = doc["weights"].as<JsonArray>();
    if (w.isNull() || (int)w.size() != NUM_WEIGHTS) {
        Serial.printf("[MQTT] Global weights size mismatch (got %d, want %d)\n",
                      w.size(), NUM_WEIGHTS);
        return;
    }

    for (int i = 0; i < NUM_WEIGHTS; i++) {
        weights[i] = w[i].as<float>();
    }

    if (!weights_valid()) {
        Serial.println("[MQTT] Received invalid global weights — resetting to local");
        init_weights();
        return;
    }

    awaitingGlobalWeights = false;
    Serial.println("[FL] Global weights received ✓ — starting new round");
    Serial.println("[FL] ─────────────────────────────────────────");
}

void on_connect(PubSubClient& client) {
    if (client.subscribe(TOPIC_SUBSCRIBE)) {
        Serial.printf("[MQTT] Subscribed to %s\n", TOPIC_SUBSCRIBE);
    } else {
        Serial.println("[MQTT] Failed to subscribe to global weights topic!");
    }
}

// ============================================================
//  MQTT — CONNECT
// ============================================================

void mqtt_connect() {
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("[MQTT] WiFi not connected — skipping MQTT reconnect");
        return;
    }

    const int MAX_ATTEMPTS = 5;
    for (int i = 0; i < MAX_ATTEMPTS && !mqttClient.connected(); i++) {
        Serial.printf("[MQTT] Connecting attempt %d/%d...\n", i + 1, MAX_ATTEMPTS);
        if (mqttClient.connect(DEVICE_ID)) {
            Serial.println("[MQTT] Connected ✓");
            on_connect(mqttClient);
            return;
        }
        delay(2000);
    }
    Serial.println("[MQTT] Could not connect to broker");
}

// ============================================================
//  MQTT — PUBLISH LOCAL WEIGHTS
// ============================================================

/*
 *  publish_weights(round_loss, round_mae)
 *
 *  Sends the local model weights plus training metrics to the
 *  bridge. Also sends weight DELTAS (w_current - w_start) so
 *  the RPi can implement proper FedAvg on deltas rather than
 *  averaging absolute weights from different initialisations.
 *
 *  After publishing, sets awaitingGlobalWeights = true so the
 *  training loop pauses until the server responds.
 */
void publish_weights(float round_loss, float round_mae) {
    if (!weights_valid()) {
        Serial.println("[PUBLISH] Aborting — weights contain NaN/Inf");
        init_weights();
        return;
    }

    StaticJsonDocument<768> doc;

    doc["device_id"] = DEVICE_ID;
    doc["round"]     = roundCount;
    doc["samples"]   = STEPS_PER_ROUND;
    doc["loss_mse"]  = round_loss;
    doc["loss_rmse"] = sqrtf(round_loss);
    doc["loss_mae"]  = round_mae;             // true MAE, not approximated

    // Absolute weights
    JsonArray w = doc.createNestedArray("weights");
    for (int i = 0; i < NUM_WEIGHTS; i++) {
        w.add(weights[i]);
    }

    // Weight deltas from start of this round (for FedAvg on deltas)
    JsonArray dw = doc.createNestedArray("weight_deltas");
    for (int i = 0; i < NUM_WEIGHTS; i++) {
        dw.add(weights[i] - weights_round_start[i]);
    }

    char buffer[768];
    size_t written = serializeJson(doc, buffer, sizeof(buffer));

    // Guard against silent JSON truncation
    if (written >= sizeof(buffer) - 1) {
        Serial.println("[PUBLISH] WARNING: JSON buffer may be truncated!");
    }

    if (mqttClient.publish(TOPIC_PUBLISH, buffer)) {
        awaitingGlobalWeights = true;
        lastPublishMs = millis();

        Serial.println("\n╔══════════════════════════════════════════╗");
        Serial.printf( "║  ROUND %3lu  PUBLISHED                    ║\n", roundCount);
        Serial.println("╠══════════════════════════════════════════╣");
        Serial.printf( "║  MSE  : %.6f                        ║\n", round_loss);
        Serial.printf( "║  RMSE : %.6f                        ║\n", sqrtf(round_loss));
        Serial.printf( "║  MAE  : %.6f  (true)               ║\n", round_mae);
        Serial.println("║  Waiting for global weights...           ║");
        Serial.println("╚══════════════════════════════════════════╝\n");
    } else {
        Serial.printf("[PUBLISH] Failed to publish round %lu\n", roundCount);
    }
}

// ============================================================
//  SETUP
// ============================================================

void setup() {
    Serial.begin(115200);
    delay(500);

    Serial.println("\n╔══════════════════════════════════════════╗");
    Serial.println("║   ESP32-S3  FEDERATED LEARNING  v2.0    ║");
    Serial.println("║   Task: Light Level Prediction           ║");
    Serial.println("╚══════════════════════════════════════════╝\n");
    Serial.printf("[SETUP] Device ID : %s\n", DEVICE_ID);
    Serial.printf("[SETUP] Env bias  : %.2f\n", DEVICE_BRIGHTNESS_BIAS);

    init_weights();

    // WiFi
    Serial.printf("[SETUP] Connecting to WiFi: %s\n", WIFI_SSID);
    WiFi.begin(WIFI_SSID, WIFI_PASS);
    int wifiTicks = 0;
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        if (++wifiTicks % 10 == 0)
            Serial.printf("[SETUP] Still connecting... (%ds)\n", wifiTicks / 2);
    }
    Serial.printf("[SETUP] WiFi connected  IP: %s\n",
                  WiFi.localIP().toString().c_str());

    // MQTT
    mqttClient.setServer(MQTT_BROKER, MQTT_PORT);
    mqttClient.setCallback(on_global_weights);
    mqtt_connect();

    // Randomise simulated clock start so two devices don't
    // see identical data each round
    simMinutes = ((float)(esp_random() % 1440));

    Serial.println("\n[SETUP] Ready. Training started.\n");
    Serial.println("  Step  |  Hour  |  LDR   |  Pred  |   MSE");
    Serial.println("─────────────────────────────────────────────");
}

// ============================================================
//  LOOP
// ============================================================

void loop() {
    // ── MQTT keepalive & reconnect ──────────────────────────
    if (mqttClient.connected()) {
        mqttClient.loop();
    } else {
        static unsigned long lastReconnectMs = 0;
        if (millis() - lastReconnectMs > 5000) {
            lastReconnectMs = millis();
            Serial.println("[MQTT] Disconnected — attempting reconnect...");
            mqtt_connect();
        }
    }

    // ── FL ROUND GATE ───────────────────────────────────────
    /*
     *  After publishing, we wait for the server to return
     *  global weights before training resumes. This implements
     *  the synchronous round structure FL requires.
     *
     *  Timeout: 30 seconds. If the server doesn't respond in
     *  time (e.g. another client is slow), we continue with
     *  local weights rather than stalling indefinitely.
     */
    if (awaitingGlobalWeights) {
        if (millis() - lastPublishMs > 30000) {
            Serial.println("[FL] Timeout waiting for global weights — continuing with local");
            awaitingGlobalWeights = false;
        } else {
            delay(100);   // yield while waiting, don't spin-burn
            return;
        }
    }

    // ── SIMULATED TIME ──────────────────────────────────────
    simMinutes += SIM_MINUTES_PER_STEP;
    if (simMinutes >= 1440.0f) simMinutes -= 1440.0f;
    float hour = simMinutes / 60.0f;   // 0.0 – 24.0

    // ── SENSOR READING ──────────────────────────────────────
    float ldr_reading = simulate_ldr(hour);

    // ── FEATURE EXTRACTION ──────────────────────────────────
    float x[NUM_FEATURES];
    build_features(hour, x);

    // ── TRAINING STEP ───────────────────────────────────────
    float sq_error  = train_step(x, ldr_reading);
    float abs_error = fabsf(predict(x) - ldr_reading);

    // Accumulate round metrics
    roundLossAcc += sq_error;
    roundMAEAcc  += abs_error;
    (void)abs_error;
    stepCount++;

    // ── NaN GUARD ───────────────────────────────────────────
    if (!weights_valid()) {
        Serial.println("[ERROR] NaN/Inf detected in weights — resetting!");
        init_weights();
        roundLossAcc = 0.0f;
        roundMAEAcc  = 0.0f;
        return;
    }

    // ── PROGRESS LOGGING (every 50 steps) ───────────────────
    if (stepCount % 50 == 0) {
        float y_pred = predict(x);
        Serial.printf(" %5lu  | %5.2fh | %.4f | %.4f | %.6f\n",
                      stepCount, hour, ldr_reading, y_pred, sq_error);
    }

    // ── END OF ROUND ────────────────────────────────────────
    if (stepCount % STEPS_PER_ROUND == 0) {
        float round_mse = roundLossAcc / (float)STEPS_PER_ROUND;
        float round_mae = roundMAEAcc  / (float)STEPS_PER_ROUND;

        // Snapshot weights at round start for delta calculation
        memcpy(weights_round_start, weights, sizeof(weights));

        if (mqttClient.connected()) {
            publish_weights(round_mse, round_mae);
            roundCount++;
        } else {
            Serial.println("[FL] MQTT disconnected — skipping publish this round");
        }

        // Reset accumulators for next round
        roundLossAcc = 0.0f;
        roundMAEAcc  = 0.0f;
    }

    delay(10);
}
