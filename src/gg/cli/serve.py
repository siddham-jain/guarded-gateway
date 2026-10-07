import argparse
import sys

import uvicorn

from gg.config.loader import ConfigError
from gg.config.settings import Settings


def register(subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    serve = subparsers.add_parser("serve", help="run the gateway with uvicorn")
    serve.add_argument("--host", help="bind address (default GG_HOST)")
    serve.add_argument("--port", type=int, help="bind port (default GG_PORT)")
    serve.add_argument("--reload", action="store_true", help="reload on code changes (dev only)")
    check = subparsers.add_parser("config", help="validate config files and print the config hash")
    check.add_argument("config_command", choices=("check",))


def run_serve(args: argparse.Namespace) -> int:
    settings = Settings()
    # one worker per container (master plan §4); scale by running more containers
    uvicorn.run(
        "gg.app.factory:build_app",
        factory=True,
        host=args.host or settings.host,
        port=args.port or settings.port,
        reload=args.reload,
        loop="uvloop",
        http="httptools",
        log_config=None,
        access_log=False,
        timeout_graceful_shutdown=int(settings.server.drain_s),
    )
    return 0


def run_config_check() -> int:
    from gg.app.factory import build_app

    try:
        app = build_app()
    except ConfigError as exc:
        for problem in exc.problems:
            print(f"error: {problem}", file=sys.stderr)
        return 1
    services = app.state.services
    print(f"config ok  hash={services.config_hash}  profile={services.settings.model_profile}")
    return 0
