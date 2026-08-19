# QGrip

QGrip is a typed Python 3.14 application for collecting EMG data, training and
validating gesture models, and safely controlling a Handi hand through the Arduino
UNO Q Router. It includes an installable FastAPI/Svelte dashboard and a completely
standalone UNO Q runtime.

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
