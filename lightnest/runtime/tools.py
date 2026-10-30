"""Typed synchronous/asynchronous tools for the Any2Any agent."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import inspect
from typing import Any, Callable


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    call_id: str = ""


@dataclass(frozen=True)
class ToolResult:
    name: str
    output: Any
    error: str | None = None
    call_id: str = ""


class ToolRegistry:
    def __init__(self): self._tools: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, function: Callable[..., Any]):
        if not name or name in self._tools or not callable(function): raise ValueError("invalid or duplicate tool")
        self._tools[name] = function

    def describe(self):
        return sorted(self._tools)

    async def call(self, request: ToolCall) -> ToolResult:
        if request.name not in self._tools: return ToolResult(request.name, None, "unknown_tool", request.call_id)
        try:
            function = self._tools[request.name]
            output = function(**request.arguments)
            if inspect.isawaitable(output): output = await output
            return ToolResult(request.name, output, call_id=request.call_id)
        except Exception as error:
            return ToolResult(request.name, None, f"{type(error).__name__}: {error}", request.call_id)

    async def call_many(self, requests: list[ToolCall]) -> list[ToolResult]:
        return list(await asyncio.gather(*(self.call(request) for request in requests)))
