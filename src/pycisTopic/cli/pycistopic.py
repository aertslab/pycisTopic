from __future__ import annotations

import argparse

from pycisTopic.cli.subcommand.tss import add_parser_tss
from pycisTopic.cli.subcommand.qc import add_parser_qc


def main():
    parser = argparse.ArgumentParser(description="pycisTopic CLI.")

    subparsers = parser.add_subparsers(
        title="Commands",
        description="List of available commands for pycisTopic CLI.",
        dest="command",
        help="Command description.",
    )
    subparsers.required = True

    add_parser_tss(subparsers)
    add_parser_qc(subparsers)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
