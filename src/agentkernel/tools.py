"""A small inspectable tool registry; writes require approval for their exact arguments."""

from __future__ import annotations

import ast
import json
import math
import operator
import os
import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentkernel.types import AgentError, ToolCall, ToolContext

ToolFunction = Callable[[ToolContext, dict[str, Any]], Awaitable[str]]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    properties: dict[str, Any]
    required: tuple[str, ...]
    handler: ToolFunction
    dangerous: bool = False

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.properties,
                    "required": list(self.required),
                    "additionalProperties": False,
                },
            },
        }


class ToolRegistry:
    def __init__(self, tools: list[Tool]) -> None:
        if len({tool.name for tool in tools}) != len(tools):
            raise ValueError("Tool names must be unique")
        self.tools = {tool.name: tool for tool in tools}

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self.tools.values()]

    async def execute(self, context: ToolContext, call: ToolCall) -> str:
        tool = self.tools.get(call.name)
        if tool is None:
            raise AgentError(f"Unknown tool: {call.name}")
        if set(call.arguments) - set(tool.properties) or set(tool.required) - set(call.arguments):
            raise AgentError("Tool arguments have missing or unsupported fields")
        for name, value in call.arguments.items():
            expected = tool.properties[name].get("type")
            if expected == "string" and not isinstance(value, str):
                raise AgentError(f"Argument {name} must be a string")
            if expected == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
                raise AgentError(f"Argument {name} must be an integer")
        if tool.dangerous and not await context.approval(call):
            raise AgentError(f"Approval denied for tool: {call.name}")
        return await tool.handler(context, call.arguments)


def safe_path(context: ToolContext, path: str) -> Path:
    relative = Path(path)
    if relative.is_absolute() or ".." in relative.parts:
        raise AgentError("Tool paths must stay inside the configured root")
    candidate = context.root
    for part in relative.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise AgentError("Tool paths cannot contain symlinks")
    if not candidate.resolve().is_relative_to(context.root):
        raise AgentError("Tool path escapes the root")
    return candidate


async def read_file(context: ToolContext, arguments: dict[str, Any]) -> str:
    path = safe_path(context, arguments["path"])
    if not path.is_file() or path.stat().st_size > context.max_file_bytes:
        raise AgentError("File missing, not regular, or exceeds size limit")
    with path.open("rb") as handle:
        content = handle.read(context.max_file_bytes + 1)
    if len(content) > context.max_file_bytes:
        raise AgentError("File grew beyond the size limit")
    return content.decode("utf-8")


async def list_files(context: ToolContext, arguments: dict[str, Any]) -> str:
    directory = safe_path(context, arguments.get("path", "."))
    if not directory.is_dir():
        raise AgentError("Directory does not exist")
    # Bound enumeration rather than silently retaining a giant directory listing.
    entries = []
    with os.scandir(directory) as items:
        for item in items:
            if len(entries) >= 1000:
                raise AgentError("Directory listing exceeds 1000 entries")
            entries.append(
                {
                    "name": item.name,
                    "kind": "symlink"
                    if item.is_symlink()
                    else "directory"
                    if item.is_dir(follow_symlinks=False)
                    else "file",
                }
            )
    return json.dumps(sorted(entries, key=lambda item: item["name"]))


async def write_file(context: ToolContext, arguments: dict[str, Any]) -> str:
    path = safe_path(context, arguments["path"])
    content = arguments["content"].encode("utf-8")
    if len(content) > context.max_file_bytes or not path.parent.is_dir():
        raise AgentError("Write exceeds size limit or parent directory is missing")
    fd, temporary = tempfile.mkstemp(prefix=".agentkernel-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return json.dumps({"written": arguments["path"], "bytes": len(content)})


async def calculate(context: ToolContext, arguments: dict[str, Any]) -> str:
    expression = arguments["expression"]
    if len(expression) > 256:
        raise AgentError("Expression is too long")
    tree = ast.parse(expression, mode="eval")
    if len(list(ast.walk(tree))) > 64:
        raise AgentError("Expression is too complex")
    operations = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Mod: operator.mod,
    }

    def evaluate(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
            result = float(node.value)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            result = evaluate(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif isinstance(node, ast.BinOp) and type(node.op) in operations:
            result = operations[type(node.op)](evaluate(node.left), evaluate(node.right))
        else:
            raise AgentError("Only numeric arithmetic + - * / % is allowed")
        if not math.isfinite(result) or abs(result) > 1e100:
            raise AgentError("Arithmetic result exceeds numeric limits")
        return result

    return json.dumps({"result": evaluate(tree.body)})


def default_registry() -> ToolRegistry:
    string = {"type": "string"}
    return ToolRegistry(
        [
            Tool(
                "read_file",
                "Read a UTF-8 file inside the allowed root",
                {"path": string},
                ("path",),
                read_file,
            ),
            Tool(
                "list_files",
                "List one directory inside the allowed root",
                {"path": string},
                (),
                list_files,
            ),
            Tool(
                "calculate",
                "Evaluate bounded numeric arithmetic",
                {"expression": string},
                ("expression",),
                calculate,
            ),
            Tool(
                "write_file",
                "Write or replace a UTF-8 file; exact arguments require approval",
                {"path": string, "content": string},
                ("path", "content"),
                write_file,
                dangerous=True,
            ),
        ]
    )
