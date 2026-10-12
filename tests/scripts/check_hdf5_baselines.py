#!/usr/bin/env python3
"""Check that GPU regression baselines were downloaded from Git LFS."""

from pathlib import Path
import sys

def main():
    directory = Path(__file__).resolve().parents[1] / "baselines"
    failed = False
    for name in ("advection_u", "advection_v", "advection_w", "stretching",
                 "twisting", "2dbubble"):
        path = directory / f"{name}.h5"
        try:
            with path.open("rb") as stream:
                header = stream.read(128)
                pointer = header.startswith(
                    b"version https://git-lfs.github.com/spec/v1")
            if pointer:
                raise ValueError("Git LFS pointer remains; baseline data was not downloaded")
            if not header.startswith(b"\x89HDF\r\n\x1a\n"):
                raise ValueError("baseline does not have the expected HDF5 signature")
        except (OSError, ValueError) as error:
            print(f"::error file=tests/baselines/{path.name}::{error}", file=sys.stderr)
            failed = True
        else:
            print(f"[OK] {path.name}: HDF5 baseline signature")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
