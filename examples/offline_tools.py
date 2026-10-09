"""Replay a plan and real local tool calls, without model inference."""

import asyncio
import json
from dataclasses import asdict
from pathlib import Path

from agentkernel.providers import ScriptedProvider
from agentkernel.runtime import Agent
from agentkernel.tools import default_registry
from agentkernel.types import ToolCall, ToolContext, Turn


async def main():
    provider = ScriptedProvider(
        [
            Turn('{"steps":["Read notes.txt","Calculate the expression","Report observed data"]}'),
            Turn(
                calls=(
                    ToolCall("read", "read_file", {"path": "notes.txt"}),
                    ToolCall("math", "calculate", {"expression": "(18 + 24) / 6"}),
                )
            ),
            Turn("Fixture replay executed read_file and calculate. Expected arithmetic result: 7."),
        ]
    )
    root = Path(__file__).parent / "fixtures"
    result = await Agent(provider, default_registry(), ToolContext(root)).run(
        "Run the offline demo"
    )
    print(json.dumps(asdict(result), indent=2))
    print("Observed tool results:")
    for message in provider.requests[-1][0]:
        if message["role"] == "tool":
            print(message["name"], message["content"])


asyncio.run(main())
