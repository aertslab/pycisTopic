from __future__ import annotations

import argparse

from pycisTopic.cli.subcommand.qc import add_parser_qc
from pycisTopic.cli.subcommand.topic_modeling import add_parser_topic_modeling
from pycisTopic.cli.subcommand.tss import add_parser_tss


def main():
    parser = argparse.ArgumentParser(description="pycisTopic CLI.")

    subparsers = parser.add_subparsers(
        title="Commands",
        description="List of available commands for pycisTopic CLI.",
        dest="command",
        help="Command description.",
    )
    subparsers.required = True

    add_parser_qc(subparsers)
    add_parser_topic_modeling(subparsers)
    add_parser_tss(subparsers)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
