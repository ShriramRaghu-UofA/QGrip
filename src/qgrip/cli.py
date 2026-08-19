"""Argparse command adapter over QGrip services."""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import threading
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from qgrip.artifacts import discover_artifacts, export_capture, latest_capture
from qgrip.assets import download_assets
from qgrip.devices import check_device, create_device
from qgrip.domain import SGTRequest, TrainingRequest
from qgrip.errors import QGripError, ValidationError
from qgrip.handi import HandiRuntime
from qgrip.profiles import default_profile, load_profile, profile_document, write_profile_atomic
from qgrip.workflows import InferenceService, SGTService, TrainingService


def _profile_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qgrip", description="EMG acquisition, training, inference, and Handi control"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    doctor = commands.add_parser("doctor", help="validate a profile and test device readiness")
    _profile_argument(doctor)

    profile = commands.add_parser("profile", help="create or inspect profiles")
    profile_commands = profile.add_subparsers(dest="profile_command", required=True)
    create = profile_commands.add_parser("create")
    create.add_argument("path", type=Path)
    create.add_argument(
        "--device", choices=["sifi", "myo_ble", "myo_dongle", "synthetic"], default="synthetic"
    )
    for name in ("validate", "show"):
        command = profile_commands.add_parser(name)
        command.add_argument("path", type=Path)

    assets = commands.add_parser("assets")
    assets_commands = assets.add_subparsers(dest="assets_command", required=True)
    download = assets_commands.add_parser("download")
    download.add_argument("--target", type=Path, default=Path("assets/gestures"))
    download.add_argument("--manifest", type=Path)

    sgt = commands.add_parser("sgt", help="run screen-guided capture")
    sgt.add_argument("subject")
    _profile_argument(sgt)
    sgt.add_argument("--discrete", action="store_true")

    export = commands.add_parser("export", help="derive Parquet from capture logs")
    export.add_argument("subject")
    export.add_argument("captures", nargs="*", type=Path)
    _profile_argument(export)

    train = commands.add_parser("train")
    train.add_argument("subject")
    _profile_argument(train)
    train.add_argument("--input", action="append", type=Path, default=[])
    train.add_argument("--model", choices=["transformer", "cnn1d", "cnn2d", "dense"])
    train.add_argument("--discrete", action="store_true")

    infer = commands.add_parser("infer")
    infer.add_argument("model", type=Path)
    _profile_argument(infer)

    web = commands.add_parser("web")
    _profile_argument(web)

    handi = commands.add_parser("handi")
    handi_commands = handi.add_subparsers(dest="handi_command", required=True)
    run = handi_commands.add_parser("run", help="standalone UNO Q runtime")
    _profile_argument(run)
    run.add_argument("--model", type=Path, required=True)
    run.add_argument("--no-api", action="store_true")
    calibrate = handi_commands.add_parser("calibrate")
    _profile_argument(calibrate)
    calibrate.add_argument("--output", type=Path, required=True)
    calibrate.add_argument("--controller")
    return parser


def _run_handi(args: argparse.Namespace) -> int:
    profile = load_profile(args.profile)
    runtime = HandiRuntime(profile, str(args.model))
    config = profile.handi
    assert config is not None
    logging.getLogger("qgrip.handi").info(
        "device=%s model=%s mapping=%s limits=%s start=%s api=%s",
        profile.device,
        runtime.model.metadata,
        dict(config.gesture_mapping),
        config.joints,
        {item.name: item.start for item in config.joints},
        "disabled" if args.no_api else f"{config.api_host}:{config.api_port}",
    )
    for event in (signal.SIGINT, signal.SIGTERM):
        signal.signal(event, lambda _signum, _frame: runtime.stop())
    if args.no_api or not config.api_enabled:
        runtime.run()
        return 0
    from qgrip.handi_api import create_handi_app

    worker = threading.Thread(target=runtime.run, name="qgrip-handi-runtime", daemon=False)
    worker.start()
    try:
        import uvicorn

        uvicorn.run(
            create_handi_app(runtime), host=config.api_host, port=config.api_port, workers=1
        )
    finally:
        runtime.stop()
        worker.join(timeout=10)
        runtime.close()
    return 0


def dispatch(args: argparse.Namespace) -> int:
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
        manifest = json.loads(args.manifest.read_text(encoding="utf-8")) if args.manifest else None
        print(
            f"downloaded {download_assets(args.target, manifest)} asset(s) "
            f"to {args.target.resolve()}"
        )
        return 0
    profile = load_profile(args.profile)
    if args.command == "doctor":
        print(json.dumps(check_device(profile.device), indent=2))
    elif args.command == "sgt":
        path = SGTService().run(
            SGTRequest(args.subject, profile, not args.discrete), threading.Event()
        )
        print(path)
    elif args.command == "export":
        captures = args.captures or [latest_capture(profile, args.subject)]
        for capture in captures:
            print(export_capture(capture))
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
        inference = InferenceService(args.model, profile.inference.backend)
        device = create_device(profile.device)
        try:
            device.connect()
            prediction = inference.predict(
                device.read(
                    max(1, int(device.sample_rate_hz * profile.inference.interval_seconds))
                ).samples
            )
            print(json.dumps(asdict(prediction), indent=2))
        finally:
            device.close()
    elif args.command == "web":
        import secrets

        import uvicorn

        from qgrip.api import create_app

        token = secrets.token_urlsafe(24)
        print(
            f"QGrip dashboard: http://{profile.dashboard.host}:{profile.dashboard.port}/?token={token}"
        )
        uvicorn.run(
            create_app(profile, token=token),
            host=profile.dashboard.host,
            port=profile.dashboard.port,
            workers=1,
        )
    elif args.command == "handi" and args.handi_command == "run":
        return _run_handi(args)
    elif args.command == "handi" and args.handi_command == "calibrate":
        if profile.handi is None:
            raise ValidationError("profile has no Handi configuration")
        print(write_profile_atomic(profile, args.output))
    else:
        print(discover_artifacts(profile))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    try:
        return dispatch(build_parser().parse_args(argv))
    except (QGripError, OSError, ValueError) as exc:
        print(f"qgrip: error: {exc}", file=sys.stderr)
        return 2
