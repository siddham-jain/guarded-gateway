import shutil
import subprocess
from pathlib import Path


class GitError(RuntimeError):
    pass


def _git(*args: str) -> subprocess.CompletedProcess[bytes]:
    git = shutil.which("git")
    if git is None:
        raise GitError("git is not on PATH")
    return subprocess.run([git, *args], capture_output=True, check=False)  # noqa: S603 - fixed argv, no shell


def read_at_ref(ref: str, path: Path) -> bytes | None:
    """file content at a commit, None if the file didn't exist there; path is relative to the repo root"""
    if _git("rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}").returncode != 0:
        raise GitError(f"unknown git ref {ref!r}")
    spec = f"{ref}:{path.as_posix()}"
    if _git("cat-file", "-e", spec).returncode != 0:
        return None
    shown = _git("show", spec)
    if shown.returncode != 0:
        raise GitError(f"git show {spec} failed: {shown.stderr.decode(errors='replace').strip()}")
    return shown.stdout
