"""Run a release build with machine-independent Rust diagnostic paths."""

import os
from pathlib import Path
import subprocess
import sys


def main(command):
    env = os.environ.copy()
    encoded = env.get("CARGO_ENCODED_RUSTFLAGS")
    flags = encoded.split("\x1f") if encoded else env.get("RUSTFLAGS", "").split()
    roots = (
        (Path.home(), "/build-home"),
        (Path(env.get("CARGO_HOME", Path.home() / ".cargo")), "/rust-deps"),
        (Path.cwd(), "/umap-alternative"),
    )
    for directory, replacement in roots:
        path = directory.resolve()
        for prefix in dict.fromkeys((str(path), path.as_posix())):
            flags.append(f"--remap-path-prefix={prefix}={replacement}")
    flags.append("-Cstrip=debuginfo")
    if sys.platform == "win32":
        flags.append("-Clink-arg=/PDBALTPATH:%_PDB%")
    env["CARGO_ENCODED_RUSTFLAGS"] = "\x1f".join(flags)
    return subprocess.run(command, env=env).returncode


if __name__ == "__main__":
    if len(sys.argv) == 1:
        raise SystemExit("Usage: python scripts/release_build.py <build command> [args...]")
    raise SystemExit(main(sys.argv[1:]))
