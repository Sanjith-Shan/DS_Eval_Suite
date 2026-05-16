"""Reference solution shim: copies regression_fixed.py to /output/regression.py."""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "regression_fixed.py"
DST = Path("/output/regression.py")


def main() -> None:
    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_text(SRC.read_text())
    print(f"Copied {SRC} -> {DST}")


if __name__ == "__main__":
    main()
