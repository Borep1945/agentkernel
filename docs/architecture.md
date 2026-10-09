# Runtime contracts

## Plan → act → observe → complete

`Planner` creates a dedicated request with no tools. The provider returns a JSON
steps list, checked for count, type and length. Planning consumes the same turn,
token and cost budgets as execution. An invalid plan ends the run before any
tool call. The validated plan is included in the execution system message.

The agent requests a `Turn` with provider-neutral text, tool calls and usage.
A no-tool turn finishes execution. Tool-call batches are checked against the
remaining tool budget before any call runs. A registry validates supported
argument fields and primitive string/integer types, requests approval for a
dangerous tool, and invokes its async handler. Successes and expected errors are
appended as tool observations with the matching call id. Models can react to a
denial; repeated requests eventually hit the turn or tool budget.

## Provider adapters

The compatible adapter calls `/chat/completions` beneath the supplied base URL,
uses function tools and extracts `usage`. Ollama calls `/api/chat` with
`stream=false`, converts JSON argument strings into native objects, and extracts
`prompt_eval_count`/`eval_count`. Missing usage or malformed provider messages
raise a sanitized error; no raw server response or credential enters logs.
The adapters use a caller-owned HTTPX client; they do not silently retry paid
requests or select a model on the caller's behalf.

`ScriptedProvider` replays fixtures and records input messages for tests. It is
never described as reasoning or inference. Wire-format tests are useful contract
checks, but endpoint/model-specific live compatibility remains to be verified by
users choosing those endpoints. Models must support tools and JSON planning.

## Limits and side effects

Provider timeout wraps each call, and asyncio cancellation propagates. There is
no separate timeout on tool handlers: built-ins are local bounded operations,
and an extension must provide its own timeout or isolation. File I/O runs on the
event loop; network filesystems may block it. A write is atomic at replacement,
not a transaction across multiple tools, and an approved replacement overwrites
the selected file. There is no automated backup for an approved write.

The runtime checks reported token and estimated cost totals after responses.
Turn and tool limits are admission checks; token/cost totals can exceed a limit
by the final request. `max_context_chars` checks serialized input before each
request. A large observation can therefore stop the next request. Provider
output limits are configured by adapters, but external services can ignore them.

Paths are relative, traversal and symlinks are rejected, and file reads/writes
have a configured byte ceiling. These checks are not a kernel sandbox. The model
receives observed file contents, so select the root according to provider access
policy. Prompt-injection text can request a tool, but a write still passes the
same explicit approval function. No approval is inferred from model text.

## Protocol references

Adapter wire formats were checked against [Ollama tool calling](https://docs.ollama.com/capabilities/tool-calling)
and the [Chat Completions API reference](https://developers.openai.com/api/reference/resources/chat).
The compatible adapter targets the function-call Chat Completions format with
`max_tokens`; endpoints that require different fields need an adapter extension.
HTTP streaming/client lifecycle follows [HTTPX async support](https://www.python-httpx.org/async/).
