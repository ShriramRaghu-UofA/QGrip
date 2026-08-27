# QGrip architecture

QGrip is one domain application with several adapters. The CLI, FastAPI dashboard,
Python API, and standalone Handi entry point do not implement parallel versions of the
workflow; they call the same typed services and device adapters.

```text
CLI ---------------------+
                         |
FastAPI dashboard API ---+--> Workflow services --> Profiles / artifacts
                         |             |
Python callers ----------+             +--> SGT / training / inference
                                       |
Svelte SPA -- HTTP/SSE ----------------+
                                       |
                                       +--> sifi-streamer public APIs
                                                |
                                                +--> SiFi bridge
                                                +--> Synthetic device
                                                +--> QGrip Myo adapter

Standalone Handi entry point
        |
        +--> LiveEMGSession --> InferenceService --> HandController
                                                     |
                                                     +--> MessagePack-RPC
                                                           |
                                                           +--> UNO Q Arduino Router
```

Dependencies point inward. Wire and command adapters depend on workflow services;
services depend on frozen domain values and narrow acquisition, artifact, model, and RPC
interfaces. Services never import the dashboard. Pydantic is confined to HTTP request
models, and untyped third-party data is converted at the boundary.

## Source map

Modules are grouped into subpackages by function:

| Module | Responsibility |
| --- | --- |
| `core/domain.py` | Frozen, slotted configuration and result values plus shared enums |
| `core/errors.py` | Stable domain errors shared by command and HTTP adapters |
| `core/profiles.py` | Strict schema-version-1 parsing, relative-path resolution, defaults, and atomic profile writes |
| `capture/streaming.py` | `sifi-streamer` configuration, device factories, Myo protocol adapter, live rolling windows, health, and prediction debounce |
| `capture/artifacts.py` | Capture inspection, strict capture-to-Parquet projection, activation-energy derivation, and artifact discovery |
| `capture/assets.py` | Explicit, checksum-verified gesture-image download |
| `capture/rpc.py` | Concurrent MessagePack-RPC client for the Arduino Router Unix socket |
| `ml/training.py` | Window construction, split/calibration logic, Torch optimization, metrics, and artifact creation |
| `ml/models.py` | Shared preprocessing, four classifiers, strict checkpoint loading, ONNX export, and ONNX Runtime adapter |
| `runtime/workflows.py` | Screen-guided capture, lazy training/inference facades, and process-local job coordination |
| `runtime/api.py` | Token-protected dashboard HTTP/SSE adapter |
| `runtime/cli.py` | Argparse adapter and foreground lifecycle ownership |
| `runtime/handi.py` | Standalone inference-to-motor runtime and safety limits |

`src/qgrip/vendor/` is third-party source and is not part of QGrip's application cleanup
surface. `src/qgrip/dashboard/` contains built frontend output packaged in the wheel;
`frontend/` is its source.

## Configuration boundary

A profile is loaded once into a frozen `QGripProfile`. Relative `data_root` and
`assets_root` values resolve beside the profile file, making behavior independent of the
launching shell's current directory. The parser rejects unknown keys, invalid ranges,
unknown enum values, duplicate gestures, profiles without `rest`, and schema versions
other than `1`.

The sections have distinct owners:

- `device` selects SiFi, Myo BLE, Myo dongle, or synthetic acquisition and declares the
  nominal channel count and sample rate. For SiFi, `sample_rate_hz` and `imu_sample_rate_hz`
  select the bridge's onboard EMG and IMU rates and are validated by `sifi-streamer` itself
  (both sensors are enabled, mirroring Myo's always-on, currently unconsumed IMU stream).
  Myo BLE and Myo dongle have a hardware-fixed `sample_rate_hz` of 200Hz and no
  `imu_sample_rate_hz` concept; either field diverging from that is rejected.
- `acquisition` maps to public `sifi-streamer` configuration and health thresholds.
- `sgt` defines gesture order, trials, preparation/practice timing, and UI update cadence.
- `model` selects one architecture and only its allowed architecture parameters.
- `training` controls windowing, STFT, activation estimation, optimization, and export.
- `inference` controls backend and CPU/GPU preference policy, cadence, confidence
  gating, and debounce.
- `dashboard` controls the local server bind address and port.
- `handi`, when present, defines the Router socket, verified joints, grip presets, step
  size, and gesture-to-action mapping.

Proportional/discrete mode is carried on each SGT and training request. The dashboard uses
`sgt.proportional` as its collection mode and requires a valid subject calibration before
enabling proportional capture. The CLI defaults to proportional unless `--discrete` is
selected.

`profile_document()` serializes the resolved typed value. `write_profile_atomic()` writes
and flushes a temporary file, atomically replaces the target, and validates the result.
The current Handi HTTP calibration endpoint can jog and read positions but deliberately
does not persist them; its save response reports `saved: false`.

## Acquisition and lifecycle ownership

QGrip consumes `sifi-streamer` only through its public acquisition APIs. Device factories
are picklable and instantiated in the streamer-owned worker. The SiFi adapter uses the TCP
bridge, synthetic acquisition uses `SyntheticSiFiDevice`, and the Myo adapter translates
the attributed transport into the same `emg_armband` stream contract.

The component that starts work owns shutdown:

- a CLI workflow owns its foreground service and context managers;
- `WorkflowCoordinator` owns one non-daemon cooperative worker thread and its cancel event;
- `LiveEMGSession` owns one `BackgroundHandle` acquisition worker for its context lifetime;
- standalone Handi owns acquisition, inference, RPC, and cleanup;
- FastAPI lifespan closes its coordinator; Uvicorn always runs with one worker.

The coordinator permits only one active job in its process. Capture and inference own
hardware; export and training share the same serialization rule so a dashboard cannot
start conflicting work while another job is still active. Cancellation sets a cooperative
event, and `close()` requests cancellation before joining the owned thread. FastAPI
`BackgroundTasks` is not used for long-running workflows.

`LiveEMGSession` reads by an absolute shared-buffer cursor. It emits only after a complete
model window exists and the requested number of fresh samples has arrived. Missing samples,
invalid validity flags, or consumer ring-buffer overruns clear the rolling window, so model
input never spans a known discontinuity. Streamer health and QGrip's consumer-overrun count
are surfaced together.

## Screen-guided capture

`SGTService` creates a unique UTC-named authoritative capture under
`<data_root>/<subject>/raw/`. Capture-level attributes include subject, creation time,
device, nominal sample rate, channel count, class order, and proportional/discrete mode.
Backend timestamps, packet sequences, capture boundaries, and SGT markers—not browser
timing—are authoritative.

For every gesture, SGT can run a preparation countdown and an unlabelled practice segment.
Proportional capture requires a subject calibration artifact containing the rest floor and
per-class robust maximum references. Recorded work is nested into trial and presentation
segments; non-rest gestures are presented at held stepped targets (25%, 50%, 75%, and
100% by default), while rest is always 0. Each presentation carries both its prompted
target and measured calibrated EMG activation. A discrete presentation is 1 for a
non-rest gesture and 0 for rest. Manual mode uses a condition-backed command gate for
pause, resume, repeat, and abort. Repeated presentations are retained in the capture log
but marked `presentation_superseded`.

The capture log is append-only and authoritative. A normal completion writes a terminal
capture-stop record. An aborted or damaged log remains evidence but cannot be exported as a
complete training dataset.

## Capture-to-Parquet projection

`export_capture()` performs one streaming pass over the capture. It selects only EMG packets
inside completed, non-superseded `presentation` segments. Preparation, practice, aborted,
and repeated presentations do not enter the training projection. Malformed channel layouts
or samples are logged and dropped. Export refuses to overwrite an existing Parquet file.

For each accepted sample, the projection records signal channels, gesture and prompted
activation, trial/order fields, device and sample-rate identity, capture provenance, host
timestamps, and lost-sample information. It also calculates `activation_energy` causally
within each presentation. For every trailing window it:

1. computes each channel's mean and mean square;
2. derives the non-negative per-channel variance;
3. averages variance across channels; and
4. takes the square root.

This is channel-demeaned RMS energy. The first samples use the available prefix rather than
future data. Schema metadata records method `causal_rms`, configured seconds, and effective
sample count. The canonical column is `sample_rate_hz`; there is no alias for older Parquet
shapes.

## Training and activation targets

Training accepts explicit capture or Parquet paths. A capture is projected automatically
when its adjacent Parquet does not yet exist. With no inputs, the latest subject capture is
selected. Every dataset must contain the profile's exact channel layout, `sample_rate_hz`,
class labels, grouping/order provenance, and—when proportional—matching activation-energy
metadata.

Windows are grouped by source capture, trial, and gesture. The normal split is a seeded
group shuffle, keeping a presentation group entirely on one side. Calibration and dataset
normalization are fitted from training indices only and then applied to both splits.

For proportional training:

1. the median energy of training rest windows becomes the rest floor;
2. each non-rest class gets its own configured energy quantile as nominal maximum;
3. activation is `(energy - rest_floor) / (class_reference - rest_floor)`, clipped to
   `[0, 1]`; and
4. rest activation remains exactly zero.

A class whose training reference does not exceed the rest floor is rejected because its
activation scale is not identifiable. The model learns both classification logits and a
sigmoid activation head. Below `activation_smoothing_threshold`, classification targets
interpolate linearly between rest and the prompted gesture; no probability is assigned to
unrelated gestures. Smooth L1 loss for the activation head is added using
`activation_loss_weight`. Discrete training uses hard class targets and no activation
loss; inference reports activation `1.0` for that model shape.

All presets share `EMGPreprocessor`, so normalization and the windowed-DFT transform are part
of the trained graph and ONNX export. `dataset_standardize` stores training-only channel statistics;
`signed_8bit` scales Myo values by 128; `window_zscore` normalizes each input window.

## Model artifact contract

Every training run gets a unique directory under `<data_root>/<subject>/models/`. The best
validation-loss state is written to `model.pt`. `metadata.json` mirrors checkpoint metadata
without weights, and `metrics.json` stores each epoch. ONNX export writes `model.onnx`; a
failed optional export leaves `onnx-error.txt` without discarding the valid Torch checkpoint.

Checkpoint loading is strict. Version `1` requires:

- `model_state_dict`, `model_name`, and canonical `model_config`;
- a non-empty ordered label list matching `model_config.n_classes`;
- device and `sample_rate_hz` identity;
- input provenance and validation metrics; and
- activation calibration metadata for proportional runs.

Window size, channel count, class count, STFT parameters, normalization, and activation-head
mode exist only in `model_config`. Missing versions, unsupported versions, unknown models,
or incomplete model configuration fail before inference. QGrip intentionally provides no
legacy checkpoint or Parquet migration layer.

## Inference

`InferenceService` always loads the adjacent `.pt` checkpoint for metadata, even when the
requested model path is `.onnx`. Backend policy is:

- `auto`: prefer adjacent ONNX; fall back to Torch if it is absent or cannot load;
- `onnx`: require ONNX and fail if it cannot load; and
- `torch`: load the checkpoint using the profile's CPU/GPU preference.

For both backends, a GPU preference selects CUDA when available and falls back to CPU;
a CPU preference forces CPU even on GPU hosts. ONNX uses `CUDAExecutionProvider` only
when it is installed. Both backends accept the
same `(samples, channels)` raw-EMG contract because preprocessing is embedded in the model.
`InferenceService.predict()` keeps a model-sized history and left-pads an incomplete direct
call with zeros. Production live paths instead wait for a complete contiguous
`LiveEMGSession` window.

Softmax produces gesture confidence. A proportional head is clipped to `[0, 1]`; a
discrete model returns `1.0`. Live CLI, dashboard, and Handi paths apply the profile's
confidence gate by converting low-confidence output to `rest`, then require consecutive
agreement through `PredictionDebouncer` before accepting a gesture switch. Dashboard and
Handi additionally reject a live stream whose nominal sample rate differs from checkpoint
metadata.

Offline benchmarking reports the backend and actual compute device alongside latency
percentiles and throughput. The dashboard benchmarks every loadable backend on CPU and
adds its GPU result only when that backend actually acquires a CUDA provider.

## Dashboard and HTTP boundary

The installed wheel contains compiled Svelte assets, so Node is not a runtime dependency.
`qgrip web` creates a random launch token and prints it in the URL. Protected API requests
send it through `X-QGrip-Token`; browser `EventSource` sends it as the SSE query parameter
because custom headers are unavailable there. Tokens use constant-time comparison. Request
bodies larger than 1 MiB are rejected.

The dashboard API exposes device probing, artifact discovery, SGT control, export, training,
inference, and offline benchmarking operations under `/api/v1`. The coordinator's latest
`JobStatus` is the
authoritative state. One authenticated SSE connection emits:

- `status`, a complete snapshot consumers must apply; and
- `notification`, an ignorable convenience event for terminal transitions.

SSE waits on the coordinator condition rather than busy-polling. The bounded wait exists to
notice disconnected clients. Domain errors use a structured HTTP 409 response; authentication
failures use 401, and request-model validation remains FastAPI's 422 response.

The dashboard has no proxy into a running standalone Handi process; the two are independent.
The Handi process owns its hardware and RPC connection exclusively while it runs.

## Standalone Handi

`qgrip-rpc-handi` is an adapter for `qgrip handi run`; it has no dashboard, browser, asset,
or sibling-process dependency. Startup validates that Handi is enabled, verified joint limits
exist, checkpoint channels and nominal sample rate match the configured device, and the Router
socket accepts a connection. Only then is the configured start pose sent.

Every requested position is clamped against the joint's verified limits, and each
`set_positions` RPC payload carries an ordered value for every configured joint. `open` and
`close` mappings apply `step * activation` to each joint sequentially; named grip mappings
apply a configured preset in one call. Unmapped predictions are no-ops. Any acquisition,
inference, or RPC exception marks the controller unhealthy and exits through exactly-once
cleanup.

Stopping QGrip prevents further movement commands and closes RPC, but it does not remove servo
torque. Software stop is not a physical emergency stop.

## Extension rules

- Add a device by adapting it to the public `sifi-streamer` device protocol; do not subclass an
  application workflow.
- Add a model by extending `ModelName`, profile architecture validation, the model factory, and
  round-trip/ONNX tests together.
- Add an HTTP or CLI feature as a thin adapter over a typed service rather than placing domain
  logic in the framework layer.
- Add artifact fields deliberately and update the strict reader, writer, metadata, tests, and
  documentation together. Do not silently accept aliases.
- Preserve capture logs as the source of truth and derive replaceable analytical artifacts from
  them.
- Keep the process that creates a hardware, worker, thread, or socket resource responsible for
  cancelling, joining, and closing it.
