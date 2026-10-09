"""Provider-neutral tool calls, usage and bounded execution configuration."""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


class AgentError(RuntimeError):
    """An agent run cannot proceed under its declared contract."""


class BudgetExceeded(AgentError):
    """A turn, tool, token or estimated-cost budget was reached."""


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Turn:
    text: str = ""
    calls: tuple[ToolCall, ...] = ()
    input_tokens: int = 0
    output_tokens: int = 0


class Provider(Protocol):
    async def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Turn: ...


@dataclass(frozen=True)
class Limits:
    max_turns: int = 8
    max_tool_calls: int = 12
    max_tokens: int = 8192
    max_context_chars: int = 32000
    max_cost_usd: float = 0.25
    input_usd_per_million: float = 0.0
    output_usd_per_million: float = 0.0
    provider_timeout: float = 30.0

    def __post_init__(self) -> None:
        for name in ("max_turns", "max_tool_calls", "max_tokens", "max_context_chars"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "max_cost_usd",
            "input_usd_per_million",
            "output_usd_per_million",
            "provider_timeout",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.provider_timeout == 0:
            raise ValueError("provider_timeout must be positive")


Approval = Callable[[ToolCall], Awaitable[bool]]


async def deny_approval(call: ToolCall) -> bool:
    return False


@dataclass
class ToolContext:
    root: Path
    approval: Approval = deny_approval
    max_file_bytes: int = 100_000

    def __post_init__(self) -> None:
        self.root = self.root.resolve(strict=True)
        if not self.root.is_dir() or self.max_file_bytes < 1:
            raise ValueError("Tool root must be a directory; max_file_bytes must be positive")


@dataclass
class Usage:
    turns: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0


@dataclass
class RunResult:
    answer: str
    plan: list[str]
    usage: Usage
    events: list[dict[str, Any]] = field(default_factory=list)
