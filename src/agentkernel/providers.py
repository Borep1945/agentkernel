"""Ollama, OpenAI-compatible HTTP and an explicitly scripted offline provider."""

from __future__ import annotations

import json
from typing import Any

import httpx

from agentkernel.types import AgentError, ToolCall, Turn


def _calls(raw: list[dict[str, Any]], *, ollama: bool = False) -> tuple[ToolCall, ...]:
    if not isinstance(raw, list):
        raise AgentError("Tool calls must be a list")
    calls = []
    for index, item in enumerate(raw):
        function = item["function"]
        arguments = function["arguments"]
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if not isinstance(arguments, dict):
            raise AgentError("Tool arguments must be a JSON object")
        name = function["name"]
        if not isinstance(name, str) or not name:
            raise AgentError("Tool name must be a nonempty string")
        call_id = item.get("id", f"ollama-{index}" if ollama else "")
        if not isinstance(call_id, str) or not call_id:
            raise AgentError("Tool call id must be a nonempty string")
        calls.append(ToolCall(call_id, name, arguments))
    if len({call.id for call in calls}) != len(calls):
        raise AgentError("Provider returned duplicate tool call ids")
    return tuple(calls)


def _usage(value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise AgentError("Provider usage must be nonnegative integers")
    return value


def _content(message: Any) -> str:
    if not isinstance(message, dict):
        raise AgentError("Provider message must be an object")
    content = message.get("content")
    if content is None:
        return ""
    if not isinstance(content, str):
        raise AgentError("Provider message content must be text")
    return content


def _endpoint(base_url: str, model: str, max_output_tokens: int) -> str:
    url = httpx.URL(base_url)
    if (
        url.scheme not in {"http", "https"}
        or not url.host
        or url.username
        or url.password
        or url.query
        or url.fragment
    ):
        raise ValueError("Provider URL must be HTTP(S) without credentials, query or fragment")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("A nonempty model name is required")
    if type(max_output_tokens) is not int or max_output_tokens < 1:
        raise ValueError("max_output_tokens must be a positive integer")
    return base_url.rstrip("/")


class OpenAICompatibleProvider:
    """An endpoint ending at /v1; pricing is supplied to the runtime separately."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        base_url: str,
        model: str,
        api_key: str = "",
        max_output_tokens: int = 1024,
    ) -> None:
        self.client = client
        self.base_url = _endpoint(base_url, model, max_output_tokens)
        self.model = model
        self.api_key = api_key
        self.max_output_tokens = max_output_tokens

    async def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Turn:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    key: value
                    for key, value in message.items()
                    if not (message.get("role") == "tool" and key == "name")
                }
                for message in messages
            ],
            "max_tokens": self.max_output_tokens,
        }
        if tools:
            payload["tools"] = tools
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            response = await self.client.post(
                f"{self.base_url}/chat/completions", headers=headers, json=payload
            )
            response.raise_for_status()
            data = response.json()
            message = data["choices"][0]["message"]
            usage = data["usage"]
            return Turn(
                _content(message),
                _calls(message.get("tool_calls", [])),
                _usage(usage["prompt_tokens"]),
                _usage(usage["completion_tokens"]),
            )
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            # Do not include raw server bodies, URLs or credentials in errors.
            raise AgentError(f"Compatible provider response failed: {type(exc).__name__}") from None


class OllamaProvider:
    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        model: str,
        base_url: str = "http://127.0.0.1:11434",
        max_output_tokens: int = 1024,
    ) -> None:
        self.client = client
        self.base_url = _endpoint(base_url, model, max_output_tokens)
        self.model = model
        self.max_output_tokens = max_output_tokens

    async def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Turn:
        converted = []
        for message in messages:
            item = {key: value for key, value in message.items() if key != "tool_call_id"}
            if item.get("role") == "tool" and "name" in item:
                item["tool_name"] = item.pop("name")
            if "tool_calls" in item:
                item["tool_calls"] = [
                    {
                        "type": "function",
                        "function": {
                            "name": call["function"]["name"],
                            "arguments": json.loads(call["function"]["arguments"]),
                        },
                    }
                    for call in item["tool_calls"]
                ]
            converted.append(item)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": converted,
            "stream": False,
            "options": {"num_predict": self.max_output_tokens},
        }
        if tools:
            payload["tools"] = tools
        try:
            response = await self.client.post(f"{self.base_url}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
            message = data["message"]
            return Turn(
                _content(message),
                _calls(message.get("tool_calls", []), ollama=True),
                _usage(data["prompt_eval_count"]),
                _usage(data["eval_count"]),
            )
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            raise AgentError(f"Ollama response failed: {type(exc).__name__}") from None


class ScriptedProvider:
    """Deterministic fixture replay. This class performs no inference or network I/O."""

    def __init__(self, turns: list[Turn]) -> None:
        self.turns = iter(turns)
        self.requests: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []

    async def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Turn:
        self.requests.append((list(messages), list(tools)))
        try:
            return next(self.turns)
        except StopIteration:
            raise AgentError("Offline fixture has no remaining turns") from None
