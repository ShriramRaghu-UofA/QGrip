"""Dedicated standalone Handi console entry point."""

import sys

from qgrip.cli import main as qgrip_main


def main() -> int:
    return qgrip_main(["handi", "run", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
