# Inference script

Simple environment inference loop for Raspberry Pi project.

Usage:

Run once:

```bash
".venv/Scripts/python.exe" inference.py --once
```

Run continuously with 10s interval:

```bash
".venv/Scripts/python.exe" inference.py --interval 10
```

Model file `model.json` must contain JSON with fields:

{
  "weights": [w1, w2],
  "bias": b,
  "threshold": 0.5
}

If `model.json` is missing, a safe default model is used.
