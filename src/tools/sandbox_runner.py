"""
Subprocess entry point for sandboxed tool execution.
Reads a JSON payload from stdin, imports module.function, calls it, writes JSON result to stdout.

Usage: python -m src.tools.sandbox_runner  (input on stdin)
"""
from __future__ import annotations

import importlib
import json
import sys


def main() -> None:
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw)
        module_path = payload["module"]
        function_name = payload["function"]
        input_data = payload.get("input", {})

        module = importlib.import_module(module_path)
        fn = getattr(module, function_name)
        result = fn(**input_data)

        if isinstance(result, str):
            sys.stdout.write(result)
        else:
            sys.stdout.write(json.dumps(result))
        sys.stdout.flush()
        sys.exit(0)
    except Exception as e:
        sys.stderr.write(str(e))
        sys.stderr.flush()
        sys.exit(1)


if __name__ == "__main__":
    main()
