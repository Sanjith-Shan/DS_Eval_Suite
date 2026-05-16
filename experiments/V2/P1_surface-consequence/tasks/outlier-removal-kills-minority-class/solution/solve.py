"""Reference solution shim: copies classify_fixed.py to /output/classify.py.

Harbor's oracle agent runs solve.sh which calls this script.
The actual fix lives next to this file as classify_fixed.py.
"""

from __future__ import annotations
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "classify_fixed.py"
DST = Path("/output/classify.py")


def main() -> None:
    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_text(SRC.read_text())
    print(f"Copied {SRC} -> {DST}")


if __name__ == "__main__":
    main()
