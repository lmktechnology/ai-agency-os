from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from src.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

_SUBPROCESS_TIMEOUT = 30  # seconds


class ToolExecutor:
    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    async def execute(self, name: str, input: dict[str, Any], sandboxed: bool = True) -> str:
        """
        Execute a tool by name with the given input dict.
        Returns a JSON string result.
        If sandboxed=True, runs in a subprocess via sandbox_runner.py.
        If sandboxed=False, runs in-process via executor thread pool.
        """
        if sandboxed:
            return await self._execute_sandboxed(name, input)
        else:
            return await self._execute_in_process(name, input)

    async def _execute_in_process(self, name: str, input: dict[str, Any]) -> str:
        fn = self._registry.get_callable(name)
        loop = asyncio.get_event_loop()
        try:
            if asyncio.iscoroutinefunction(fn):
                result = await fn(**input)
            else:
                result = await loop.run_in_executor(None, lambda: fn(**input))
            return json.dumps(result) if not isinstance(result, str) else result
        except Exception as e:
            logger.error("Tool %s failed: %s", name, e)
            return json.dumps({"error": str(e)})

    async def _execute_sandboxed(self, name: str, input: dict[str, Any]) -> str:
        config = self._registry.get_config(name)
        payload = json.dumps({
            "module": config.module,
            "function": config.function,
            "input": input,
        })

        try:
            proc = await asyncio.create_subprocess_exec(
                "python", "-m", "src.tools.sandbox_runner",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=payload.encode()),
                timeout=_SUBPROCESS_TIMEOUT,
            )
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except Exception:
                pass
            return json.dumps({"error": f"Tool {name} timed out after {_SUBPROCESS_TIMEOUT}s"})
        except Exception as e:
            return json.dumps({"error": f"Subprocess launch failed: {e}"})

        if proc.returncode != 0:
            err = stderr.decode().strip()
            logger.error("Sandboxed tool %s exited with %d: %s", name, proc.returncode, err)
            return json.dumps({"error": err or f"Tool exited with code {proc.returncode}"})

        return stdout.decode().strip()
