import argparse
import getpass
import sys
from datetime import UTC, datetime
from typing import Any, TextIO

import yaml
from pydantic import ValidationError

from gg.auth.config import KeysConfig
from gg.auth.keys import generate_key, hash_key, is_well_formed


def register(subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    keys = subparsers.add_parser("keys", help="manage virtual keys")
    commands = keys.add_subparsers(dest="keys_command", required=True)
    create = commands.add_parser("create", help="mint a key; prints the token once and a keys.yaml entry")
    create.add_argument("--id", required=True, dest="key_id", help="stable key id, e.g. demo")
    create.add_argument("--name", required=True, help="human-readable name")
    create.add_argument("--env", choices=("live", "test"), default="live")
    commands.add_parser("hash", help="print the keys.yaml hash of a token read from stdin")


def _create(args: argparse.Namespace, stdout: TextIO, stderr: TextIO) -> int:
    generated = generate_key(args.env)
    entry: dict[str, Any] = {
        "id": args.key_id,
        "name": args.name,
        "hash": generated.hash,
        "prefix": generated.prefix,
        "created_at": datetime.now(UTC).date(),
    }
    try:
        KeysConfig.model_validate({"keys": [entry]})
    except ValidationError as exc:
        for err in exc.errors(include_input=False):
            field = ".".join(str(p) for p in err["loc"][2:]) or "entry"
            print(f"error: {field}: {err['msg']}", file=stderr)
        return 1
    print(generated.token, file=stdout)
    print("this token is shown once; store it now. add this entry under `keys:` in keys.yaml:", file=stderr)
    print(yaml.safe_dump([entry], sort_keys=False).rstrip(), file=stderr)
    return 0


def _hash(stdin: TextIO, stdout: TextIO, stderr: TextIO) -> int:
    token = (getpass.getpass("token: ") if stdin.isatty() else stdin.readline()).strip()
    if not is_well_formed(token):
        print("error: not a well-formed gg key", file=stderr)
        return 1
    print(hash_key(token), file=stdout)
    return 0


def run(
    args: argparse.Namespace,
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    out, err = stdout or sys.stdout, stderr or sys.stderr
    if args.keys_command == "create":
        return _create(args, out, err)
    return _hash(stdin or sys.stdin, out, err)
