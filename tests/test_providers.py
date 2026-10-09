import httpx
import pytest

from agentkernel.providers import OllamaProvider, OpenAICompatibleProvider
from agentkernel.types import AgentError


async def test_compatible_request_and_tool_parsing():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "function": {
                                        "name": "calculate",
                                        "arguments": '{"expression":"2+2"}',
                                    },
                                }
                            ],
                        }
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            client, base_url="https://fixture.test/v1", model="fixture", api_key="fixture-key"
        )
        turn = await provider.complete([{"role": "user", "content": "test"}], [])
    assert turn.calls[0].arguments == {"expression": "2+2"}
    assert turn.input_tokens == 10
    assert requests[0].url.path == "/v1/chat/completions"
    assert requests[0].headers["Authorization"] == "Bearer fixture-key"


async def test_ollama_request_and_usage():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "message": {
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "list_files", "arguments": {"path": "."}}}
                    ],
                },
                "prompt_eval_count": 20,
                "eval_count": 5,
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        turn = await OllamaProvider(client, model="fixture").complete([], [])
    assert turn.calls[0].id == "ollama-0"
    assert turn.output_tokens == 5
    assert requests[0].url.path == "/api/chat"


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"choices": []},
        {"choices": [{"message": {}}]},
        {
            "choices": [{"message": {"content": "x"}}],
            "usage": {"prompt_tokens": -1, "completion_tokens": 2},
        },
    ],
)
async def test_malformed_compatible_response(data):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=data))
    ) as client:
        with pytest.raises(AgentError):
            await OpenAICompatibleProvider(
                client, base_url="https://fixture.test", model="fixture"
            ).complete([], [])


async def test_provider_error_does_not_expose_server_body_or_key():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(401, text="private response body")
        )
    ) as client:
        with pytest.raises(AgentError) as error:
            await OpenAICompatibleProvider(
                client, base_url="https://fixture.test", model="fixture", api_key="private-key"
            ).complete([], [])
    assert "private" not in str(error.value)


@pytest.mark.parametrize(
    "message", [[], "bad", {"content": ["bad"]}, {"content": "x", "tool_calls": "bad"}]
)
async def test_wrong_message_shapes_are_sanitized(message):
    data = {
        "choices": [{"message": message}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=data))
    ) as client:
        with pytest.raises(AgentError):
            await OpenAICompatibleProvider(
                client, base_url="https://fixture.test", model="fixture"
            ).complete([], [])


@pytest.mark.parametrize(
    "endpoint",
    ["file:///etc/passwd", "https://user:password@host.test", "https://host.test?token=secret"],
)
def test_invalid_endpoint_rejected(endpoint):
    with pytest.raises(ValueError):
        OpenAICompatibleProvider(httpx.AsyncClient(), base_url=endpoint, model="fixture")


async def test_ollama_tool_observation_uses_native_tool_name():
    import json

    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200, json={"message": {"content": "done"}, "prompt_eval_count": 1, "eval_count": 1}
        )

    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "a",
                    "type": "function",
                    "function": {"name": "list_files", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "a", "name": "list_files", "content": "[]"},
    ]
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await OllamaProvider(client, model="fixture").complete(messages, [])
    assert requests[0]["messages"][1] == {
        "role": "tool",
        "tool_name": "list_files",
        "content": "[]",
    }
    assert requests[0]["messages"][0]["tool_calls"][0]["function"]["arguments"] == {}
