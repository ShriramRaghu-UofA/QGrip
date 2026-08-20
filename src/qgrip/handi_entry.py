"""Dedicated standalone Handi console entry point."""

import sys

from qgrip.cli import main as qgrip_main


def main() -> int:
    """Expose the Handi-only console script without importing dashboard components."""
    return qgrip_main(["handi", "run", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
