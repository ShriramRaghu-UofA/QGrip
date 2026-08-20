# Contributing to QGrip

Python 3.14 or newer and `uv` are mandatory. Use `uv` for environments, dependency
changes, locking, commands, tests, and builds. Add packages with `uv add`; never edit
`uv.lock` manually. Ruff and ty are required development dependencies. Every
QGrip-authored Python file must pass both tools without directory-wide exclusions.

Node is required only for work in `frontend/`. Installed users receive compiled assets in
the wheel. Use `npm ci`, do not edit generated assets, and never commit `node_modules`.
Behavior changes require regression tests; tests that generate artifacts must use temporary
directories. Do not commit unless explicitly requested.

`src/qgrip/vendor/` is attributed third-party source. Leave it byte-identical for ordinary
QGrip work; a deliberate vendor update must be isolated, provenance-reviewed, and committed
separately. Exclude vendor code from mechanical formatting and application-wide cleanup.

Read `AGENTS.md` and `ARCHITECTURE.md` before changing lifecycle, public APIs, profiles,
artifacts, inference, or hardware control.

## Required validation

```powershell
uv sync --locked --all-extras --dev
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run python -m unittest discover -s tests -v

Push-Location frontend
npm ci
npm run check
npm run test
npm run build
Pop-Location

git diff --check
uv build
```

Before release, install the wheel into a clean Python 3.14 environment and smoke-test CLI
help, packaged dashboard assets, `qgrip-rpc-handi`, and a synthetic workflow.
