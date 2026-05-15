"""Reference solution shim: copies the corrected pipeline to /output/.

The actual fix lives next to this file as `pipeline_fixed.py`. Harbor's oracle
agent simply needs `/output/pipeline_fixed.py` to exist for the verifier to
import. We copy rather than symlink so the test environment is self-contained.
"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "pipeline_fixed.py"
DST = Path("/output/pipeline_fixed.py")


def main() -> None:
    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_text(SRC.read_text())


if __name__ == "__main__":
    main()
