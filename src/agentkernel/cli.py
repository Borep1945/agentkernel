from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import tomllib
from dataclasses import asdict
from pathlib import Path

import httpx

from agentkernel.providers import OllamaProvider, OpenAICompatibleProvider, ScriptedProvider
from agentkernel.runtime import Agent
from agentkernel.tools import default_registry
from agentkernel.types import AgentError, Limits, ToolCall, ToolContext, Turn, deny_approval


async def confirm(call: ToolCall) -> bool:
    print("Tool requests a write:", json.dumps(asdict(call), indent=2))
    try:
        answer = await asyncio.to_thread(input, "Type APPROVE to execute these exact arguments: ")
    except EOFError:
        return False
    return answer == "APPROVE"


async def run(args: argparse.Namespace) -> int:
    settings = tomllib.loads(args.config.read_text()) if args.config else {}
    if set(settings) - {"limits"}:
        raise ValueError("Unknown configuration section")
    limits = Limits(**settings.get("limits", {}))
    context = ToolContext(
        args.root,
        approval=confirm if args.interactive_approval else deny_approval,
    )
    async with httpx.AsyncClient(timeout=limits.provider_timeout, trust_env=False) as client:
        if args.provider == "offline":
            provider = ScriptedProvider(
                [
                    Turn('{"steps":["List the selected directory","Report the offline demo"]}'),
                    Turn(calls=(ToolCall("fixture-list", "list_files", {"path": "."}),)),
                    Turn(
                        "Offline fixture completed: list_files executed. No AI inference occurred."
                    ),
                ]
            )
        elif args.provider == "ollama":
            if not args.model:
                raise ValueError("--model is required for a live provider")
            provider = OllamaProvider(
                client, model=args.model, base_url=args.base_url or "http://127.0.0.1:11434"
            )
        else:
            if not args.model or not args.base_url:
                raise ValueError("--model and --base-url are required for a compatible provider")
            provider = OpenAICompatibleProvider(
                client,
                model=args.model,
                base_url=args.base_url,
                api_key=os.getenv("AGENTKERNEL_API_KEY", ""),
            )
        result = await Agent(provider, default_registry(), context, limits).run(args.goal)
    print(json.dumps(asdict(result), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bounded tool agent; defaults to an offline fixture"
    )
    parser.add_argument("goal", nargs="?", default="List the selected directory")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--provider", choices=["offline", "ollama", "compatible"], default="offline"
    )
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--interactive-approval", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s"
    )
    try:
        return asyncio.run(run(args))
    except (AgentError, OSError, ValueError, TypeError) as exc:
        parser.exit(2, f"agentkernel: {exc}\n")
    return 2
