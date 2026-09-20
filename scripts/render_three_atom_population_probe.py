"""Embed the three-atom diagnostic JSON into the inline HTML fragment."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("template", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    template = args.template.read_text(encoding="utf-8")
    payload = args.data.read_text(encoding="utf-8").replace("</", "<\\/")
    if template.count("__DATA__") != 1:
        raise RuntimeError("template must contain exactly one __DATA__ marker")
    args.output.write_text(template.replace("__DATA__", payload), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
