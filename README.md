# QGrip

QGrip is a typed Python 3.14 application for collecting EMG data, training and
validating gesture models, and safely controlling a Handi hand through the Arduino
UNO Q Router. It includes an installable FastAPI/Svelte dashboard and a completely
standalone UNO Q runtime.

QGrip has three ways to use the same workflow services:

- the local web dashboard for guided capture, training, and live validation;
- the `qgrip` command-line interface for scripted or terminal-based workflows; and
- Python services for applications that need to consume real-time predictions directly.

The capture log is the source of truth. Parquet datasets, model checkpoints, ONNX
exports, metadata, and metrics are derived artifacts written under the profile's
`data_root`.

## Quick start

```powershell
uv sync --locked --all-extras --dev
uv run qgrip profile create synthetic.json --device synthetic
uv run qgrip doctor --profile synthetic.json
uv run qgrip sgt demo --profile synthetic.json
uv run qgrip export demo --profile synthetic.json
uv run qgrip train demo --profile synthetic.json
uv run qgrip web --profile synthetic.json
```

The synthetic profile is useful for checking the software path. For a physical
device, create the appropriate `sifi`, `myo_ble`, or `myo_dongle` profile and edit
its device settings before running `doctor`.

## Choose a workflow

Use the dashboard when an operator needs guided collection, artifact selection, and
a live validation display. Use the CLI when the workflow is being scripted or run
without a browser. Use the Python API when another application needs to receive
predictions in-process. All three use the profile, typed workflow services, and
device adapters described below.

Only one hardware-owning activity may run in a process. Do not run a CLI inference
loop, dashboard inference job, Handi runtime, or another `LiveEMGSession` at the
same time against the same process/device.

## Launching the web dashboard

From a source checkout, install the locked environment and create or select a
profile, then start the server:

```powershell
uv sync --locked --all-extras --dev
uv run qgrip profile create synthetic.json --device synthetic
uv run qgrip web --profile synthetic.json
```

QGrip prints a launch URL similar to:

```text
QGrip dashboard: http://127.0.0.1:8765/?token=...
```

Open the complete printed URL in a browser. The per-launch token in that URL is
required by the dashboard API, so opening `http://127.0.0.1:8765/` without it will
not connect. Keep the terminal running while using the dashboard and press
`Ctrl+C` to stop it.

For an installed wheel, omit `uv run`:

```powershell
qgrip web --profile C:\path\to\profile.json
```

The bind address and port come from the profile's `dashboard` section. The
templates default to loopback access on port 8765:

```json
{
  "dashboard": {
    "host": "127.0.0.1",
    "port": 8765
  }
}
```

The compiled dashboard is included in the Python wheel; Node and the frontend
source tree are not required to run it. Use Node only when developing or rebuilding
the frontend.

The dashboard guides the complete workflow:

1. **Setup** validates the selected profile and acquisition device.
2. **Collect** runs SGT and preserves the authoritative JSONL capture.
3. **Export** derives the canonical training Parquet from that capture.
4. **Train** fits the selected Torch model and writes `model.pt`, `model.onnx`,
   `metadata.json`, and `metrics.json` under `data/<subject>/models/<run-id>/`.
5. **Validate** starts live acquisition and displays the predicted gesture,
   confidence, proportional activation, latency, and prediction history.

### Dashboard workflow, step by step

1. Create and validate a profile with `qgrip profile create` and `qgrip doctor`.
2. Start `qgrip web --profile <profile.json>` and open the entire URL printed by
   QGrip, including its token.
3. In **Setup**, enter a subject identifier and use **Connect device** to verify the
   same live acquisition path used by capture and inference.
4. In **Collect**, start screen-guided training (SGT). Follow each prompt; choose
   automatic or manual advancement as appropriate. QGrip writes an authoritative
   JSONL capture log.
5. Export the completed capture to Parquet from the collection screen.
6. In **Train**, select a model preset and training data, then wait for the
   checkpoint and derived artifacts to be produced.
7. In **Validate**, select the generated `.pt` checkpoint and start live inference.
   The screen shows the accepted gesture, confidence, activation, model latency,
   signal health, and a prediction history. Stop the job before leaving the
   hardware to another workflow.

The dashboard API is token-protected. It starts live inference with
`POST /api/v1/inference/start`, reports the latest accepted prediction through
`GET /api/v1/inference/status`, and also publishes status updates through the
authenticated server-sent-events endpoint `/api/v1/stream?token=...`. It is designed
for controlling and observing live acquisition; it does not accept arbitrary EMG
sample windows for one-off prediction.

## Command-line workflow

Run `uv run qgrip --help` to see the installed commands. From a source checkout,
the following is the normal end-to-end flow. Replace `synthetic` with a real device
profile and `demo` with your subject identifier when appropriate.

```powershell
# Create, inspect, and validate a profile.
uv run qgrip profile create synthetic.json --device synthetic
uv run qgrip profile validate synthetic.json
uv run qgrip doctor --profile synthetic.json

# Record a screen-guided session, then derive its training dataset.
uv run qgrip sgt demo --profile synthetic.json
uv run qgrip export demo --profile synthetic.json

# Train using the latest capture for the subject.
uv run qgrip train demo --profile synthetic.json
```

`sgt` records proportional targets by default; add `--discrete` to record and train
a discrete model instead. `export` uses the subject's latest capture by default, or
accepts one or more explicit capture-log paths. `train` likewise uses the latest
capture when `--input` is omitted; repeat `--input <dataset.parquet>` to select
specific exported datasets, and use `--model transformer|cnn1d|cnn2d|dense` to
override the profile default.

Training creates a run directory under
`data/<subject>/models/<run-id>/` containing `model.pt`, `model.onnx` when ONNX
export is enabled, `metadata.json`, and `metrics.json`. These artifacts are
self-describing and are not overwritten.

### Run live inference from the CLI

Pass the generated checkpoint to `infer`:

```powershell
# Print one accepted prediction as JSON and exit.
uv run qgrip infer data/demo/models/RUN/model.pt --profile synthetic.json --once

# Keep printing accepted predictions until Ctrl+C.
uv run qgrip infer data/demo/models/RUN/model.pt --profile synthetic.json
```

Each emitted JSON object has `gesture`, `confidence`, `activation`, and
`latency_ms`. The CLI checks model/device channel compatibility, uses the profile's
inference cadence, confidence gate, and debounce setting, and prints only accepted
predictions. Inference uses an adjacent ONNX artifact automatically when available;
otherwise it uses the Torch checkpoint.

## Use real-time inference from Python

HTTP is not required to access predictions. Create one `LiveEMGSession`, use its
rolling windows as input to `InferenceService`, and consume the returned
`Prediction` values in your own loop. The session owns the streamer acquisition
worker for the lifetime of the `with` block.

```python
import time
from dataclasses import replace

from qgrip.profiles import load_profile
from qgrip.streaming import LiveEMGSession, PredictionDebouncer, sample_rates_match
from qgrip.workflows import InferenceService

profile = load_profile("profile.json")
model = InferenceService("data/demo/models/RUN/model.pt", profile.inference.backend)

with LiveEMGSession(profile.device, profile.acquisition) as session:
    if session.channels != model.channels:
        raise ValueError("model and live stream have different channel counts")
    if not sample_rates_match(session.sample_rate_hz, float(model.metadata["sample_rate_hz"])):
        raise ValueError("model and live stream have different sample rates")

    minimum_new_samples = max(
        1, round(session.sample_rate_hz * profile.inference.inference_period_seconds)
    )
    debouncer = PredictionDebouncer(profile.inference.switch_predictions)

    while True:
        samples = session.next_window(model.window_size, minimum_new_samples)
        if samples is None:
            time.sleep(profile.inference.idle_poll_seconds)
            continue

        prediction = model.predict(samples)
        if prediction.confidence < profile.inference.confidence_gate:
            prediction = replace(prediction, gesture="rest")
        accepted = debouncer.accept(prediction)
        if accepted is not None:
            print(accepted.gesture, accepted.confidence, accepted.activation)
```

`InferenceService.predict()` returns a `Prediction` with `gesture`, `confidence`,
`activation`, and `latency_ms`. It maintains its own model-sized history, but callers
should normally use `LiveEMGSession.next_window()` because it ensures a valid,
contiguous rolling EMG window and resets after device gaps or consumer overruns.
The loop above applies the same confidence gate and debounce behavior as CLI and
dashboard live inference; omit those two steps if an application intentionally
needs every raw model prediction.

For a one-off, already-acquired window, instantiate `InferenceService` and call
`predict()` directly with a tuple of per-sample, per-channel float tuples. A `.pt`
checkpoint is the usual input. A `.onnx` path is also accepted, but its matching
`.pt` checkpoint must be present for the self-describing metadata.

Training is implemented directly in QGrip. The `transformer`, `cnn1d`, `cnn2d`,
and `dense` presets share an `EMGPreprocessor` module that owns normalization and
`torch.stft`. Proportional models learn an activation head; discrete models expose
the same inference contract with activation fixed at `1.0`. Inference automatically
uses the adjacent ONNX model when available and otherwise uses the Torch checkpoint.
No scripts or Python modules are loaded from the reference acquisition repository at
runtime.

Packaged templates for `synthetic`, `sifi`, `myo_ble`, and `myo_dongle` live in
`src/qgrip/profile_templates`. Myo support uses the attributed PyoMyo source vendored
from the reference acquisition repository; PyoMyo itself is not installed.

## Profiles

Profiles are the editable configuration boundary for QGrip. They compose device,
acquisition, SGT, model, training, inference, dashboard, and optional Handi settings;
the CLI, dashboard, and standalone Handi runtime use the same loaded profile. Create
one from a template with `qgrip profile create`, then edit it before running a workflow.
Each service reads only its own nested section: SGT reads `sgt` and `acquisition`,
training reads `training` and `model`, and live inference/Handi read `inference` and
`acquisition` alongside the device and model artifacts.

The `training` section controls every training parameter. Its three timing settings
are deliberately independent: `dataset_stride_seconds` controls overlap between
training examples, `stft_hop_samples` controls STFT frame overlap, and
`inference_period_seconds` in the separate `inference` section controls live output
rate.

```json
{
  "training": {
    "epochs": 30,
    "batch_size": 128,
    "learning_rate": 0.0001,
    "validation_fraction": 0.2,
    "training_window_seconds": 1.0,
    "dataset_stride_seconds": 0.005,
    "stft_n_fft": null,
    "stft_hop_samples": null,
    "activation_energy_window_seconds": 0.1,
    "activation_reference_quantile": 0.9,
    "activation_smoothing_threshold": 0.25,
    "activation_loss_weight": 1.0,
    "weight_decay": 0.0001,
    "normalization": "dataset_standardize",
    "seed": 42,
    "export_onnx": true
  }
}
```

Set `stft_n_fft` and `stft_hop_samples` together to override QGrip's sample-rate
dependent STFT defaults; use `null` for automatic selection. `normalization` is
`signed_8bit` for Myo profiles and `dataset_standardize` for SiFi and other devices;
the latter fits statistics from the training split for proportional and discrete models.
Profile validation
rejects unknown keys and invalid ranges before capture, training, inference, or Handi
control begins.

Every exported Parquet row contains `activation_energy`, a causal, channel-demeaned RMS
value over the trailing `activation_energy_window_seconds`; the same method and effective
sample count are stored in Parquet schema metadata. Proportional training uses the energy
on the final row of each model window. After trials are split, the rest median and each gesture's
`activation_reference_quantile` are fitted from training trials only and then applied
to both splits. This avoids leaking validation signal statistics. Rest is fixed at zero,
and gesture activation is clipped between the rest floor and its class reference.
Below `activation_smoothing_threshold`, the classification target moves linearly from
rest to the prompted gesture; unrelated gestures receive no target probability. The
prompted `activation` remains in the derived Parquet data for comparison, and fitted
energy references are recorded in model metadata.

`acquisition` is the shared `sifi-streamer` policy: shared-buffer duration, worker
acknowledgement timeout, capture flush/compression/durability behavior, and nested
signal-health thresholds. `model.architecture` accepts only parameters for the selected
model (for example, Transformer `d_model`, `nhead`, `dim_feedforward`, and `dropout`).
`inference` owns output cadence, confidence/debounce policy, and its short cooperative
waits. `dashboard.handi_timeout_seconds` controls only the optional Handi health and
calibration HTTP proxy. The shipped templates spell out all of these values so profiles
are self-contained and safe to modify.

## Standalone Handi

```powershell
uv run qgrip-rpc-handi --profile handi.json --model data/demo/models/RUN/model.pt --no-api
```

The standalone command owns acquisition, inference, the Unix-domain Arduino Router
RPC connection, start pose, movement, and shutdown. An optional loopback observer API
can be enabled in the profile. The App Lab Brick/sketch lives separately; configure
its repository in `qgrip.handi.HANDI_BRICK_REPOSITORY_URL` when published.

Every motor command is clamped to configured joint limits and sent as one
`set_positions` Router call. Stopping QGrip commands does not disable servo torque and
does not replace a physical emergency stop.

## Gesture images

QGrip does not download assets during installation, import, or capture. Download the
seven default gesture images (`rest`, `close`, `open`, wrist flexion/extension, pronation,
and supination) into `assets/images` explicitly:

```powershell
uv run qgrip assets download
```

To download no more than the classes configured in a profile and place them in that
profile's `assets_root`, run:

```powershell
uv run qgrip assets download --profile sifi.json
```

Individual classes can instead be selected by repeating `--gesture`, and `--target`
overrides the output directory:

```powershell
uv run qgrip assets download --gesture rest --gesture open --target assets/images
```

The images are checksum-verified against pinned files from
[LibEMGGestures](https://github.com/LibEMG/LibEMGGestures/tree/c17792f1966f23f7dafda7c47a65012e47a2e7ee).
That project asks image users to cite its work; follow the citation instructions in its
README and review its attribution and licensing before redistribution. QGrip writes the
source revision, checksums, and citation reminder to `manifest.json` beside the images.
Text cues remain fully usable without images.

See [CONTRIBUTING.md](CONTRIBUTING.md) and [ARCHITECTURE.md](ARCHITECTURE.md) before
changing public workflows or hardware lifecycle behavior.
