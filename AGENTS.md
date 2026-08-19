# Agent invariants

- Read `CONTRIBUTING.md` and `ARCHITECTURE.md` before editing. Stay within QGrip unless
  another repository is explicitly in scope. Do not commit unless explicitly requested.
- Consume `sifi-streamer` through public APIs. Prefer composition and narrow protocols to
  application inheritance. CLI, FastAPI, and Svelte adapt the same typed services.
- Standalone Handi must not depend on the dashboard, browser, assets, or another QGrip
  process. Never invoke migrated scripts through subprocesses.
- The component starting capture, training, inference, or RPC owns orderly shutdown. Only
  one hardware-owning operation may run in a process. Never use FastAPI `BackgroundTasks`
  for these workflows; coordinators own cooperative worker threads. Uvicorn uses one worker.
- Capture logs are authoritative; Parquet is derived. Validate profiles, devices, datasets,
  and checkpoints before work. Backend timing and capture markers are authoritative in SGT.
- Clamp all Handi commands to verified limits. Calibration atomically replaces a profile and
  never modifies Python constants.
- Prefer frozen, slotted dataclasses for fixed domain values. Pydantic belongs at HTTP wire
  boundaries. Contain `Any` at JSON, MessagePack, PyTorch, and third-party edges.
- Prefer daisyUI and Tailwind utilities to handwritten component CSS.
