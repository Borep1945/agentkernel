# Local verification

Verified on 2026-10-09 with Python 3.14.4 on macOS arm64.
CI declares Python 3.11, 3.12 and 3.13; those remote matrix jobs were not run in this local check.

| Check | Local result |
| --- | --- |
| `check .` | PASS |
| `format --check .` | PASS |
| `-m pytest -q` | PASS: 44 passed in 0.12s |
| `-m agentkernel --help` | PASS |
| `examples/offline_tools.py` | PASS |
| `-m build --no-isolation` | PASS |

## Actual offline demo output

```text
{
  "answer": "Fixture replay executed read_file and calculate. Expected arithmetic result: 7.",
  "plan": [
    "Read notes.txt",
    "Calculate the expression",
    "Report observed data"
  ],
  "usage": {
    "turns": 3,
    "tool_calls": 2,
    "input_tokens": 0,
    "output_tokens": 0,
    "estimated_cost_usd": 0.0
  },
  "events": [
    {
      "event": "provider_turn",
      "turn": 1,
      "input_tokens": 0,
      "output_tokens": 0
    },
    {
      "event": "planned",
      "steps": 3
    },
    {
      "event": "provider_turn",
      "turn": 2,
      "input_tokens": 0,
      "output_tokens": 0
    },
    {
      "event": "tool_call",
      "name": "read_file",
      "outcome": "ok",
      "index": 1
    },
    {
      "event": "tool_call",
      "name": "calculate",
      "outcome": "ok",
      "index": 2
    },
    {
      "event": "provider_turn",
      "turn": 3,
      "input_tokens": 0,
      "output_tokens": 0
    },
    {
      "event": "completed",
      "turns": 3,
      "tool_calls": 2,
      "input_tokens": 0,
      "output_tokens": 0,
      "estimated_cost_usd": 0.0
    }
  ]
}
Observed tool results:
read_file Action items: review the tests, inspect the tool root, document provider limits.

calculate {"result": 7.0}
```

The demo uses real local tools or a mock HTTP transport. It does not verify any live service or model.

## Dependency advisory check

`pip-audit --local --progress-spinner off --format json` returned exit 0
on the shared verification environment: 59 package records and no known
vulnerabilities in the audited packages. The initial bootstrap pip 26.1 had
known advisories; upgrading it to 26.2.1 resolved them without ignores.
Unpublished local AgentKernel 0.1.0 was skipped by the advisory service.
This check covers installed dependency versions; it is not an application security audit.
