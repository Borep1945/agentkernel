# Python API reference

Python 3.11+ is required. Public components are intentionally explicit: a caller
selects the provider, tool registry, allowed root, approval function and limits.
This is an in-process runtime, not an OS sandbox or a persistent agent service.

## Provider-neutral types (`agentkernel.types`)

- `ToolCall(id: str, name: str, arguments: dict[str, Any])`: frozen provider-neutral
  tool request. Ids must be unique within a returned batch.
- `Turn(text="", calls=(), input_tokens=0, output_tokens=0)`: frozen provider output.
  Token counts must be nonnegative integers. `calls` is a tuple of `ToolCall`.
- `Provider`: protocol with `async complete(messages: list[dict[str, Any]],
  tools: list[dict[str, Any]]) -> Turn`.
- `AgentError`: provider/runtime contract failure. `BudgetExceeded` subclasses it.
- `Usage`: mutable counters for turns, tool calls, input/output tokens and
  estimated cost in USD.
- `RunResult(answer: str, plan: list[str], usage: Usage, events=[])`: completed
  answer plus validated planning steps, usage and a metadata event trail.

`Limits(max_turns=8, max_tool_calls=12, max_tokens=8192,
max_context_chars=32000, max_cost_usd=0.25, input_usd_per_million=0.0,
output_usd_per_million=0.0, provider_timeout=30.0)` is a frozen configuration.
Counts are positive integers; prices and cost limits are finite nonnegative
numbers; the provider timeout is positive. Pricing defaults to zero for
local/offline use. It must be set from your selected paid provider's actual rates.

`ToolContext(root: Path, approval=deny_approval, max_file_bytes=100_000)` resolves
an existing directory root and requires a positive file-size ceiling. `approval`
is `async (ToolCall) -> bool`, called separately for every dangerous tool request.
The default returns False. Exact argument review should be part of any real
approval implementation; model text is never an approval signal.

## Agent and planner (`agentkernel.runtime`)

`Agent(provider: Provider, registry: ToolRegistry, context: ToolContext,
limits: Limits | None = None)` stores the execution dependencies.

`await agent.run(goal: str) -> RunResult` accepts a nonempty goal up to 4,000
characters. It first requests a JSON plan without tools, validates it, and then
executes provider-selected tool calls. Tool errors become observations so the
provider can respond to a failed read or denied write. An assistant turn without
tool calls finishes the run. No automatic inference of task success occurs.

`Planner.messages(goal, registry) -> list[dict[str, Any]]` prepares planning
messages with the available tool names. `Planner.parse(turn: Turn) -> list[str]`
requires JSON `{"steps":[...]}` with 1–8 nonempty strings, each at most 500
characters, and rejects tool calls during planning.

Provider calls have a per-call timeout. Async cancellation propagates to the
provider call. There is no separate timeout for custom tool handlers; extensions
must provide their own cancellation, timeout or process isolation. Built-in local
file I/O runs on the event loop and can block on a slow filesystem.

Turn count and whole tool-call batches are checked before execution. Serialized
input messages and tool schemas must fit `max_context_chars` before a request.
Reported token totals and estimated cost are checked **after** each response;
they stop later work but cannot prevent the last paid request from crossing the
threshold. A large tool observation can prevent the next provider request.

```python
import asyncio
from pathlib import Path
from agentkernel.providers import ScriptedProvider
from agentkernel.runtime import Agent
from agentkernel.tools import default_registry
from agentkernel.types import ToolCall, ToolContext, Turn


async def demo():
    provider = ScriptedProvider(
        [
            Turn('{"steps":["Calculate the expression","Report it"]}'),
            Turn(calls=(ToolCall("math", "calculate", {"expression": "2 + 2"}),)),
            Turn("Offline fixture executed calculate."),
        ]
    )
    agent = Agent(provider, default_registry(), ToolContext(Path.cwd()))
    return await agent.run("Calculate 2 + 2")


print(asyncio.run(demo()).answer)
```

This example is a fixture replay using a real local calculator tool. It performs
no model inference. The scripted answer is fixed; observed tool outputs can be
inspected in `provider.requests`.

## Providers (`agentkernel.providers`)

`OpenAICompatibleProvider(client: httpx.AsyncClient, *, base_url: str, model: str,
api_key="", max_output_tokens=1024)` calls `/chat/completions` beneath the supplied
HTTP(S) base URL, typically ending in `/v1`. It sends `max_tokens`, function tools
and bearer authentication when a key is present. URLs cannot contain credentials,
query strings or fragments. The adapter requires nonnegative integer usage fields
`prompt_tokens` and `completion_tokens`; malformed replies fail with `AgentError`.
Compatibility is limited to this Chat Completions wire format.

`OllamaProvider(client: httpx.AsyncClient, *, model: str,
base_url="http://127.0.0.1:11434", max_output_tokens=1024)` calls `/api/chat` with
`stream=false` and `num_predict`. It converts tool arguments and observations to
Ollama's native `tool_name` format and requires `prompt_eval_count`/`eval_count`.
The selected installed model must support tools and JSON planning.

Both providers use a caller-owned HTTPX client, make no silent retries, sanitize
HTTP error messages and do not close the client. The runtime supplies the timeout
wrapper; direct adapter callers must configure their own HTTPX deadline. Token
output limits are requested from the provider, which may ignore them.

`ScriptedProvider(turns: list[Turn])` replays a finite sequence, performs no network
or inference, records calls in `requests`, and raises `AgentError` when exhausted.

## Tools (`agentkernel.tools`)

`Tool(name, description, properties, required, handler, dangerous=False)` is a
frozen registration. `handler` has signature
`async (context: ToolContext, arguments: dict[str, Any]) -> str`.
`tool.schema() -> dict[str, Any]` produces a function tool schema with required
fields and `additionalProperties=False`.

`ToolRegistry(tools: list[Tool])` rejects duplicate names.
`registry.schemas() -> list[dict[str, Any]]` exports schemas.
`await registry.execute(context, call) -> str` rejects unknown tools, missing or
extra fields, and invalid supported primitive string/integer arguments. This is
not complete JSON Schema validation for arbitrary extensions. Dangerous tools
require approval before their handler runs.

`default_registry()` registers:

| Tool | Arguments | Result / boundary |
| --- | --- | --- |
| `list_files` | Optional `path` string, default `.` | JSON list of sorted entries; max 1,000 entries |
| `read_file` | Required `path` string | UTF-8 text; regular file inside root, byte limit |
| `calculate` | Required `expression` string | JSON numeric result; + - * / %, unary signs, max 256 chars and 64 AST nodes |
| `write_file` | Required `path`, `content` strings | JSON written path and bytes; approval, byte limit, existing parent |

File paths are relative; traversal and descendant symlinks are rejected. Approved
writes use atomic replacement and can overwrite an existing file. They have no
backup and are not a transaction across multiple tool calls. These path checks
are vulnerable to a concurrent process replacing ancestor directories; the root
is not a race-proof sandbox. File observations are sent to the selected provider,
so choose a root containing only data that provider is authorized to receive.

Metadata events include tool names/outcomes and usage, not arguments or contents.
`RunResult.answer` can still contain model output or private observed data; the
CLI prints it. See [architecture](architecture.md) and
[verification](VERIFICATION.md) for operational limits and current evidence.
