"""Write requirements.txt: the runtime deps from pyproject.toml, fully pinned.

Resolution targets the container (CPython 3.11, manylinux x86_64) rather than
the host, so running this on Windows or macOS yields the same file.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def main() -> int:
    with (ROOT / "pyproject.toml").open("rb") as f:
        deps = tomllib.load(f)["project"]["dependencies"]
    with tempfile.TemporaryDirectory() as tmp:
        report = Path(tmp) / "report.json"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                "--dry-run",
                "--ignore-installed",
                "--only-binary=:all:",
                "--platform",
                "manylinux_2_17_x86_64",
                "--python-version",
                "3.11",
                "--implementation",
                "cp",
                "--target",
                str(Path(tmp) / "target"),
                "--report",
                str(report),
                *deps,
            ],
            check=True,
        )
        installed = json.loads(report.read_text(encoding="utf-8"))["install"]
    pins = sorted(
        f"{_norm(i['metadata']['name'])}=={i['metadata']['version']}" for i in installed
    )
    header = "# Runtime lock for the container image. Regenerate: make lock\n"
    (ROOT / "requirements.txt").write_text(
        header + "\n".join(pins) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(pins)} pins to requirements.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
