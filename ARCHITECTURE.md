# QGrip architecture

```text
CLI -------------------+
                       +-- Typed workflow services -- Profiles / Artifacts
FastAPI dashboard API -+              |
                                      +-- SGT / Training / Inference
Svelte SPA -- same-origin HTTP -------+
                                      +-- Device adapters

Standalone Handi CLI
        |
        +-- Device acquisition
        +-- Model inference
        +-- Hand controller
        +-- Arduino Router RPC
                  |
                  +-- Optional FastAPI health/calibration adapter

Dashboard backend -- HTTP proxy -- Optional UNO Q Handi API
```

Dependencies point inward: wire and command adapters depend on typed services, which depend
on immutable domain values and narrow device/artifact protocols. Services never import the
dashboard. The creator of capture, training, inference, or RPC owns its shutdown.

Profiles are versioned, immutable runtime boundaries; relative paths resolve beside the
profile and calibration writes an atomic validated replacement. Raw capture logs are
authoritative. Parquet, checkpoints, ONNX, metrics, and metadata are derived, self-describing
artifacts and are never overwritten. Jobs are process-owned cooperative threads, with one
hardware owner at a time.

The wheel contains compiled dashboard assets, so Node is not a runtime dependency. The
loopback dashboard uses a launch token and proxies optional UNO Q HTTP requests.

Motor control validates the whole startup chain before applying a start pose, clamps each
joint to verified limits, treats unmapped predictions as no-ops, and stops movement on signal
or RPC failure. Stopping commands does not remove servo torque and is not a physical
emergency stop.
