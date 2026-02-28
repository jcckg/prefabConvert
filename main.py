#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    src_path = Path(__file__).resolve().parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    import convert  # type: ignore

    return convert.main()


if __name__ == "__main__":
    raise SystemExit(main())
