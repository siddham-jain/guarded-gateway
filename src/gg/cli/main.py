import argparse
from collections.abc import Sequence

from gg import __version__
from gg.cli import keys_cmd, serve


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gg", description="GG LLM gateway")
    parser.add_argument("--version", action="version", version=f"gg {__version__}")
    subparsers = parser.add_subparsers(dest="command")
    keys_cmd.register(subparsers)
    serve.register(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "keys":
        return keys_cmd.run(args)
    if args.command == "serve":
        return serve.run_serve(args)
    if args.command == "config":
        return serve.run_config_check()
    return 0
