"""Run the baseplate design standalone: `python -m baseplate_design`.

The normal path is automatic -- `gravity_design.py` designs the baseplates as
soon as the member optimization finishes. This entry point covers the other
case: re-designing from a `baseplate_inputs.json` that was already written,
without re-running the FE model. With no arguments it falls back to the
hand-entered single-plate demo in baseplate_design.py.

    python -m baseplate_design                                  # demo
    python -m baseplate_design output/baseplate_inputs.json      # re-design
    python -m baseplate_design output/baseplate_inputs.json out.json
"""
import json
import sys
from pathlib import Path

from .baseplate_design import design
from .config import BaseplateConfig
from .export import write_baseplate_design_json
from .uniform_design import design_uniform_baseplate


def main(argv: list[str]) -> int:
    if not argv:
        design()
        return 0

    inputs_path = Path(argv[0])
    if not inputs_path.is_file():
        print(f"No such baseplate_inputs file: {inputs_path}", file=sys.stderr)
        return 1

    inputs = json.loads(inputs_path.read_text(encoding="utf-8"))
    result = design_uniform_baseplate(inputs, BaseplateConfig())
    print(result.summary())

    out_path = Path(argv[1]) if len(argv) > 1 else \
        inputs_path.with_name("baseplate_design.json")
    print(f"\nBaseplate design written to "
          f"{write_baseplate_design_json(result, out_path)}")
    return 0 if result.feasible else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
