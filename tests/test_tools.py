import json

import pytest

from agentkernel.tools import default_registry
from agentkernel.types import AgentError, ToolCall, ToolContext


async def test_read_list_and_calculate(tmp_path):
    (tmp_path / "note.txt").write_text("observed content")
    context = ToolContext(tmp_path)
    registry = default_registry()
    assert (
        await registry.execute(context, ToolCall("r", "read_file", {"path": "note.txt"}))
        == "observed content"
    )
    listing = json.loads(await registry.execute(context, ToolCall("l", "list_files", {})))
    assert listing == [{"name": "note.txt", "kind": "file"}]
    result = json.loads(
        await registry.execute(context, ToolCall("c", "calculate", {"expression": "-3 + 10 / 2"}))
    )
    assert result["result"] == 2


@pytest.mark.parametrize("path", ["../outside", "/etc/passwd"])
async def test_escape_rejected(tmp_path, path):
    with pytest.raises(AgentError, match="root"):
        await default_registry().execute(
            ToolContext(tmp_path), ToolCall("r", "read_file", {"path": path})
        )


async def test_symlink_and_file_size_rejected(tmp_path):
    file = tmp_path / "big"
    file.write_text("123456")
    (tmp_path / "link").symlink_to(file)
    registry = default_registry()
    context = ToolContext(tmp_path, max_file_bytes=3)
    for name in ("big", "link"):
        with pytest.raises(AgentError):
            await registry.execute(context, ToolCall("r", "read_file", {"path": name}))


@pytest.mark.parametrize("expression", ["__import__('os')", "2 ** 10000", "True", "1e200"])
async def test_calculator_rejects_non_arithmetic_and_extreme_values(tmp_path, expression):
    with pytest.raises(AgentError):
        await default_registry().execute(
            ToolContext(tmp_path), ToolCall("c", "calculate", {"expression": expression})
        )


@pytest.mark.parametrize("arguments", [{}, {"path": 123}, {"path": "x", "extra": 1}])
async def test_arguments_validated_before_execution(tmp_path, arguments):
    with pytest.raises(AgentError):
        await default_registry().execute(
            ToolContext(tmp_path), ToolCall("r", "read_file", arguments)
        )
