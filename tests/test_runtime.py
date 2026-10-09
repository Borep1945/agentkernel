import asyncio
import json

import pytest

from agentkernel.providers import ScriptedProvider
from agentkernel.runtime import Agent, Planner
from agentkernel.tools import default_registry
from agentkernel.types import AgentError, BudgetExceeded, Limits, ToolCall, ToolContext, Turn


def planning():
    return Turn(
        '{"steps":["Calculate the requested expression","Report the result"]}',
        input_tokens=10,
        output_tokens=10,
    )


def fixture_agent(tmp_path, turns, limits=None, approval=None):
    context = ToolContext(tmp_path)
    if approval:
        context.approval = approval
    provider = ScriptedProvider(turns)
    return Agent(provider, default_registry(), context, limits), provider


async def test_plan_tool_observe_final_and_token_usage(tmp_path):
    agent, provider = fixture_agent(
        tmp_path,
        [
            planning(),
            Turn(
                calls=(ToolCall("math", "calculate", {"expression": "(2 + 3) * 4"}),),
                input_tokens=20,
                output_tokens=5,
            ),
            Turn("20", input_tokens=30, output_tokens=2),
        ],
    )
    result = await agent.run("Calculate (2 + 3) * 4")
    assert result.answer == "20" and result.usage.turns == 3
    assert result.usage.tool_calls == 1 and result.usage.input_tokens == 60
    assert json.loads(provider.requests[-1][0][-1]["content"]) == {"result": 20.0}
    assert [event["event"] for event in result.events] == [
        "provider_turn",
        "planned",
        "provider_turn",
        "tool_call",
        "provider_turn",
        "completed",
    ]


async def test_dangerous_tool_denied_by_default_then_observed(tmp_path):
    agent, provider = fixture_agent(
        tmp_path,
        [
            planning(),
            Turn(
                calls=(ToolCall("write", "write_file", {"path": "note.txt", "content": "private"}),)
            ),
            Turn("Write was denied"),
        ],
    )
    result = await agent.run("Write a note")
    assert not (tmp_path / "note.txt").exists()
    assert "Approval denied" in provider.requests[-1][0][-1]["content"]
    assert result.events[3]["outcome"] == "error"


async def test_exact_arguments_approved_then_atomic_write(tmp_path):
    approvals = []

    async def approve(call):
        approvals.append(call)
        return call.arguments == {"path": "note.txt", "content": "hello"}

    agent, _ = fixture_agent(
        tmp_path,
        [
            planning(),
            Turn(
                calls=(ToolCall("write", "write_file", {"path": "note.txt", "content": "hello"}),)
            ),
            Turn("done"),
        ],
        approval=approve,
    )
    await agent.run("Write a note")
    assert (tmp_path / "note.txt").read_text() == "hello"
    assert len(approvals) == 1
    assert not list(tmp_path.glob(".agentkernel-*"))


@pytest.mark.parametrize(
    "settings",
    [
        {"max_turns": 1},
        {"max_tokens": 1},
        {"max_context_chars": 10},
        {"max_cost_usd": 0.001, "input_usd_per_million": 1000},
    ],
)
async def test_budgets_enforced(tmp_path, settings):
    agent, _ = fixture_agent(tmp_path, [planning(), Turn("done")], Limits(**settings))
    with pytest.raises(BudgetExceeded):
        await agent.run("bounded task")


async def test_tool_budget_checks_whole_batch_before_any_side_effect(tmp_path):
    agent, _ = fixture_agent(
        tmp_path,
        [
            planning(),
            Turn(
                calls=(
                    ToolCall("a", "calculate", {"expression": "1"}),
                    ToolCall("b", "calculate", {"expression": "2"}),
                )
            ),
        ],
        Limits(max_tool_calls=1),
    )
    with pytest.raises(BudgetExceeded, match="max_tool_calls"):
        await agent.run("Calculate")


async def test_provider_timeout_and_cancellation(tmp_path):
    class SlowProvider:
        async def complete(self, messages, tools):
            await asyncio.Event().wait()

    agent = Agent(
        SlowProvider(), default_registry(), ToolContext(tmp_path), Limits(provider_timeout=0.01)
    )
    with pytest.raises(AgentError, match="timeout"):
        await agent.run("Do a task")


async def test_unknown_tool_is_observed_and_logs_have_no_arguments(tmp_path, caplog):
    agent, provider = fixture_agent(
        tmp_path,
        [
            planning(),
            Turn(calls=(ToolCall("unknown", "missing_tool", {"secret": "private payload"}),)),
            Turn("done"),
        ],
    )
    with caplog.at_level("INFO"):
        result = await agent.run("Do a task")
    assert "Unknown tool" in provider.requests[-1][0][-1]["content"]
    assert "private payload" not in caplog.text
    assert result.answer == "done"


@pytest.mark.parametrize(
    "turn",
    [
        Turn("plain text"),
        Turn('{"steps":[]}'),
        Turn('{"steps":[3]}'),
        Turn(calls=(ToolCall("a", "calculate", {}),)),
    ],
)
def test_invalid_plan_rejected(turn):
    with pytest.raises(AgentError):
        Planner.parse(turn)


@pytest.mark.parametrize(
    "settings",
    [
        {"max_turns": 0},
        {"max_tokens": True},
        {"provider_timeout": 0},
        {"max_cost_usd": float("nan")},
    ],
)
def test_invalid_limits(settings):
    with pytest.raises(ValueError):
        Limits(**settings)
