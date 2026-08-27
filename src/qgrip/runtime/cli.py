"""Argparse command adapter over QGrip services."""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path

from qgrip.capture.artifacts import discover_artifacts, export_capture, latest_capture
from qgrip.capture.assets import (
    DEFAULT_GESTURES,
    GESTURE_ASSETS,
    LIBEMG_CITATION_URL,
    download_assets,
)
from qgrip.capture.rpc import MessagePackRpcClient
from qgrip.capture.streaming import LiveEMGSession, PredictionDebouncer, check_streamer_device
from qgrip.core.domain import BenchmarkResult, ComputePreference, SGTRequest, TrainingRequest
from qgrip.core.errors import QGripError, ValidationError
from qgrip.core.profiles import (
    default_profile,
    load_profile,
    profile_document,
    write_profile_atomic,
)
from qgrip.runtime.handi import HandiRuntime, run_interactive
from qgrip.runtime.hid import LABEL_TO_AXIS, HidRuntime
from qgrip.runtime.workflows import (
    CalibrationService,
    InferenceService,
    SGTService,
    TrainingService,
    run_inference_benchmark,
)

DEFAULT_PROFILE_DIR = Path("data/profiles")


def _profile_path(value: str) -> Path:
    """Resolve a bare profile filename against ``data/profiles``, leaving explicit paths alone."""
    path = Path(value)
    return DEFAULT_PROFILE_DIR / path if path.parent == Path(".") else path


def _profile_argument(parser: argparse.ArgumentParser) -> None:
    """Add the required profile-file argument shared by profile-aware commands."""
    parser.add_argument(
        "--profile",
        type=_profile_path,
        required=True,
        help="profile filename or path; bare filenames resolve under data/profiles",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the complete command-line grammar without running a workflow."""
    parser = argparse.ArgumentParser(
        prog="qgrip", description="EMG acquisition, training, inference, and Handi control"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    doctor = commands.add_parser("doctor", help="validate a profile and test device readiness")
    _profile_argument(doctor)

    profile = commands.add_parser("profile", help="create or inspect profiles")
    profile_commands = profile.add_subparsers(dest="profile_command", required=True)
    create = profile_commands.add_parser("create", help="write a new default profile to disk")
    create.add_argument(
        "path",
        type=_profile_path,
        help="profile filename or path; bare filenames are created under data/profiles",
    )
    create.add_argument(
        "--device",
        choices=["sifi", "myo_ble", "myo_dongle", "synthetic"],
        default="synthetic",
        help="EMG source the new profile targets (default: synthetic)",
    )
    validate = profile_commands.add_parser(
        "validate", help="load a profile and report whether it is well-formed"
    )
    validate.add_argument("path", type=_profile_path, help="profile filename or path")
    show = profile_commands.add_parser("show", help="print a profile's resolved JSON document")
    show.add_argument("path", type=_profile_path, help="profile filename or path")

    assets = commands.add_parser("assets", help="manage optional gesture images")
    assets_commands = assets.add_subparsers(dest="assets_command", required=True)
    download = assets_commands.add_parser("download", help="download LibEMG gesture images")
    download.add_argument(
        "--target",
        type=Path,
        help="directory to save images into (default: profile's assets_root or assets/images)",
    )
    download.add_argument(
        "--profile",
        type=_profile_path,
        help="download only the gestures configured by this profile",
    )
    download.add_argument(
        "--gesture",
        action="append",
        choices=tuple(GESTURE_ASSETS),
        help="gesture to download; repeat to select multiple gestures "
        "(default: profile's gestures, or the built-in default set)",
    )

    sgt = commands.add_parser("sgt", help="run screen-guided capture")
    sgt.add_argument("subject", help="subject identifier; names the capture's output directory")
    _profile_argument(sgt)
    sgt.add_argument(
        "--discrete",
        action="store_true",
        help="present only rest/hold prompts instead of proportional activation levels",
    )

    calibration = commands.add_parser("sgt-calibrate", help="calibrate proportional SGT activation")
    calibration.add_argument(
        "subject", help="subject identifier; names the calibration's output file"
    )
    _profile_argument(calibration)

    export = commands.add_parser("export", help="derive Parquet from capture logs")
    export.add_argument("subject", help="subject identifier whose captures are exported")
    export.add_argument(
        "captures",
        nargs="*",
        type=Path,
        help="capture log path(s) to export (default: the subject's latest capture)",
    )
    _profile_argument(export)

    train = commands.add_parser("train", help="train a classifier from exported capture data")
    train.add_argument("subject", help="subject identifier whose exported data is used")
    _profile_argument(train)
    train.add_argument(
        "--input",
        action="append",
        type=Path,
        default=[],
        help="Parquet file to include; repeat to combine multiple exports "
        "(default: the subject's latest export)",
    )
    train.add_argument(
        "--model",
        choices=["transformer", "cnn1d", "cnn2d", "dense"],
        help="model architecture to train (default: profile's model.name)",
    )
    train.add_argument(
        "--discrete",
        action="store_true",
        help="train on rest/hold labels only, without a proportional activation target",
    )

    infer = commands.add_parser("infer", help="run live inference against a streaming EMG device")
    infer.add_argument(
        "model", type=Path, help="path to a trained model checkpoint (.pt or .onnx)"
    )
    _profile_argument(infer)
    infer.add_argument(
        "--once", action="store_true", help="print one accepted prediction and exit"
    )

    benchmark = commands.add_parser(
        "benchmark",
        help="measure inference latency/throughput on synthetic data (no device needed)",
    )
    benchmark.add_argument(
        "model", type=Path, help="path to a trained model checkpoint (.pt or .onnx)"
    )
    benchmark.add_argument(
        "--backend",
        choices=["auto", "torch", "onnx"],
        default="auto",
        help="inference backend to benchmark (default: auto, prefers an adjacent ONNX artifact)",
    )
    benchmark.add_argument(
        "--device",
        choices=tuple(ComputePreference),
        default=ComputePreference.GPU,
        help="preferred compute device; GPU falls back to CPU when unavailable (default: gpu)",
    )
    benchmark.add_argument(
        "--iterations",
        type=int,
        default=200,
        help="number of timed predictions to run (default: 200)",
    )
    benchmark.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="untimed predictions run first to prime backend setup (default: 20)",
    )
    benchmark.add_argument(
        "--seed", type=int, default=0, help="RNG seed for synthetic input windows (default: 0)"
    )
    benchmark.add_argument(
        "--json", action="store_true", help="print machine-readable JSON instead of a text summary"
    )

    web = commands.add_parser("web", help="launch the local dashboard web server")
    _profile_argument(web)

    handi = commands.add_parser("handi", help="standalone Handi robotic-hand runtime and setup")
    handi_commands = handi.add_subparsers(dest="handi_command", required=True)
    run = handi_commands.add_parser(
        "run", help="run inference and drive Handi continuously until stopped"
    )
    _profile_argument(run)
    run.add_argument(
        "--model",
        type=Path,
        required=True,
        help="path to a trained model checkpoint (.pt or .onnx)",
    )
    calibrate = handi_commands.add_parser(
        "calibrate", help="verify or interactively edit a profile's Handi joint limits and grips"
    )
    _profile_argument(calibrate)
    calibrate.add_argument(
        "--output",
        type=_profile_path,
        required=True,
        help="path to write the calibrated profile to; bare filenames resolve under data/profiles",
    )
    calibrate.add_argument(
        "--controller",
        help="controller identifier to use instead of the profile's configured one",
    )
    calibrate.add_argument(
        "--interactive",
        action="store_true",
        help="launch a curses wizard to jog joints to their extend/flex endpoints "
        "and edit grip presets by hand, instead of just re-validating the profile",
    )

    hid = commands.add_parser(
        "hid",
        help="run inference and drive a USB HID joystick axis continuously until stopped",
    )
    _profile_argument(hid)
    hid.add_argument(
        "--model",
        type=Path,
        required=True,
        help="path to a trained model checkpoint (.pt or .onnx)",
    )
    hid.add_argument(
        "--device",
        type=Path,
        default=Path("/dev/hidg1"),
        help="HID gadget device node to write joystick reports to (default: /dev/hidg1)",
    )
    return parser


def _run_handi(args: argparse.Namespace) -> int:
    """Own the foreground standalone-Handi runtime and signal-driven shutdown."""
    profile = load_profile(args.profile)
    runtime = HandiRuntime(profile, str(args.model))
    config = profile.handi
    assert config is not None
    logging.getLogger("qgrip.runtime.handi").info(
        "device=%s model=%s labels=%s mapping=%s limits=%s start=%s",
        profile.device,
        runtime.model.metadata.get("model_name"),
        runtime.model.labels,
        dict(config.gesture_mapping),
        config.joints,
        {item.name: item.minimum for item in config.joints},
    )
    for event in (signal.SIGINT, signal.SIGTERM):
        signal.signal(event, lambda _signum, _frame: runtime.stop())
    runtime.run()
    return 0


def _run_hid(args: argparse.Namespace) -> int:
    """Own the foreground HID-joystick runtime and signal-driven shutdown."""
    profile = load_profile(args.profile)
    runtime = HidRuntime(profile, str(args.model), device=args.device)
    logging.getLogger("qgrip.runtime.hid").info(
        "device=%s model=%s labels=%s axis_map=%s hid=%s",
        profile.device,
        runtime.model.metadata.get("model_name"),
        runtime.model.labels,
        dict(LABEL_TO_AXIS),
        args.device,
    )
    for event in (signal.SIGINT, signal.SIGTERM):
        signal.signal(event, lambda _signum, _frame: runtime.stop())
    runtime.run()
    return 0


def _print_benchmark_result(result: BenchmarkResult, *, as_json: bool) -> None:
    """Print a completed inference benchmark as JSON or an aligned text summary."""
    if as_json:
        print(json.dumps(asdict(result), indent=2))
        return
    print(f"model:          {result.model_name}")
    print(f"backend:        {result.backend}")
    print(f"device:         {result.device}")
    print(f"window shape:   ({result.window_size}, {result.channels})")
    print(f"iterations:     {result.iterations} (+{result.warmup} warmup)")
    print(f"latency mean:   {result.mean_ms:.3f} ms")
    print(f"latency median: {result.median_ms:.3f} ms")
    print(f"latency p95:    {result.p95_ms:.3f} ms")
    print(f"latency p99:    {result.p99_ms:.3f} ms")
    print(f"latency min:    {result.min_ms:.3f} ms")
    print(f"latency max:    {result.max_ms:.3f} ms")
    print(f"latency stdev:  {result.stdev_ms:.3f} ms")
    print(f"throughput:     {result.throughput_hz:.1f} predictions/sec")


def dispatch(args: argparse.Namespace) -> int:
    """Adapt parsed CLI arguments to the shared typed services and print results."""
    if args.command == "profile":
        if args.profile_command == "create":
            args.path.parent.mkdir(parents=True, exist_ok=True)
            args.path.write_text(
                json.dumps(default_profile(args.device), indent=2) + "\n", encoding="utf-8"
            )
            load_profile(args.path)
            print(args.path.resolve())
        else:
            profile = load_profile(args.path)
            print(
                json.dumps(profile_document(profile), indent=2, default=str)
                if args.profile_command == "show"
                else f"valid: {profile.path}"
            )
        return 0
    if args.command == "assets":
        asset_profile = load_profile(args.profile) if args.profile else None
        target = args.target or (
            asset_profile.assets_root if asset_profile else Path("assets/images")
        )
        gestures = args.gesture or (
            asset_profile.sgt.gestures if asset_profile else DEFAULT_GESTURES
        )
        count = download_assets(target, gestures)
        print(f"ready: {count} gesture image(s) in {target.resolve()}")
        print(f"LibEMGGestures requests citation; see {LIBEMG_CITATION_URL}")
        return 0
    if args.command == "benchmark":
        inference = InferenceService(args.model, args.backend, args.device)
        result = run_inference_benchmark(
            inference, iterations=args.iterations, warmup=args.warmup, seed=args.seed
        )
        _print_benchmark_result(result, as_json=args.json)
        return 0
    profile = load_profile(args.profile)
    if args.command == "doctor":
        print(json.dumps(check_streamer_device(profile.device, profile.acquisition), indent=2))
    elif args.command == "sgt":
        path = SGTService().run(
            SGTRequest(args.subject, profile, not args.discrete), threading.Event()
        )
        print(path)
    elif args.command == "sgt-calibrate":
        print(CalibrationService().run(args.subject, profile, threading.Event()))
    elif args.command == "export":
        captures = args.captures or [latest_capture(profile, args.subject)]
        for capture in captures:
            print(
                export_capture(
                    capture,
                    activation_energy_window_seconds=(
                        profile.training.activation_energy_window_seconds
                    ),
                )
            )
    elif args.command == "train":
        request = TrainingRequest(
            args.subject,
            profile,
            tuple(path.resolve() for path in args.input),
            args.model or profile.model.name,
            not args.discrete,
        )
        print(TrainingService().train(request, threading.Event()))
    elif args.command == "infer":
        inference = InferenceService(
            args.model, profile.inference.backend, profile.inference.device_preference
        )
        with LiveEMGSession(profile.device, profile.acquisition) as session:
            if session.channels != inference.channels:
                raise ValidationError("model channel count does not match live EMG stream")
            minimum_new_samples = max(
                1,
                round(session.sample_rate_hz * profile.inference.inference_period_seconds),
            )
            debouncer = PredictionDebouncer(profile.inference.switch_predictions)
            next_inference_at = time.monotonic()
            while True:
                wait_seconds = next_inference_at - time.monotonic()
                if wait_seconds > 0:
                    time.sleep(min(wait_seconds, profile.inference.maximum_wait_seconds))
                    continue
                samples = session.next_window(inference.window_size, minimum_new_samples)
                if samples is None:
                    time.sleep(profile.inference.idle_poll_seconds)
                    continue
                prediction = inference.predict(samples)
                if prediction.confidence < profile.inference.confidence_gate:
                    prediction = replace(prediction, gesture="rest")
                accepted = debouncer.accept(prediction)
                if accepted is not None:
                    print(json.dumps(asdict(accepted), indent=2), flush=True)
                    if args.once:
                        break
                next_inference_at += profile.inference.inference_period_seconds
                if next_inference_at < time.monotonic():
                    next_inference_at = time.monotonic()
    elif args.command == "web":
        import secrets

        import uvicorn

        from qgrip.runtime.api import create_app

        token = secrets.token_urlsafe(24)
        print(
            f"QGrip dashboard: http://{profile.dashboard.host}:{profile.dashboard.port}/?token={token}"
        )
        uvicorn.run(
            create_app(profile, token=token),
            host=profile.dashboard.host,
            port=profile.dashboard.port,
            workers=1,
            timeout_graceful_shutdown=5,
        )
    elif args.command == "handi" and args.handi_command == "run":
        return _run_handi(args)
    elif args.command == "hid":
        return _run_hid(args)
    elif args.command == "handi" and args.handi_command == "calibrate":
        if profile.handi is None:
            raise ValidationError("profile has no Handi configuration")
        if args.interactive:
            rpc = MessagePackRpcClient(profile.handi.rpc_socket, profile.handi.rpc_timeout_seconds)
            run_interactive(profile, rpc, args.output)
        else:
            print(write_profile_atomic(profile, args.output))
    else:
        print(discover_artifacts(profile))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run QGrip's console entry point and translate expected failures to exit codes."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    try:
        return dispatch(build_parser().parse_args(argv))
    except KeyboardInterrupt:
        return 130
    except (QGripError, OSError, ValueError) as exc:
        print(f"qgrip: error: {exc}", file=sys.stderr)
        return 2
