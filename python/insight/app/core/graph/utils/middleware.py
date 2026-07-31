from dataclasses import dataclass
from typing import Literal

from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
    ToolRetryMiddleware,
    wrap_model_call,
)


@dataclass(frozen=True)
class AgentExecutionLimits:
    model_calls: int
    tool_calls: int
    tool_retries: int = 0
    model_limit_behavior: Literal["end", "error"] = "end"
    tool_limit_behavior: Literal["continue", "end", "error"] = "continue"


def _force_final_answer(model_calls: int):
    """Take the tools away on the last permitted model call.

    Hitting the model-call limit with tools still bound ends the run wherever it
    happens to be — usually with no structured result at all, which then costs an
    extra coercion call to recover. Spending the final call on an answer instead
    is the same budget with something to show for it.
    """
    remaining = {"calls": model_calls}

    @wrap_model_call
    async def force_final_answer(request, handler):
        remaining["calls"] -= 1
        if remaining["calls"] <= 0:
            request = request.override(tools=[])
        return await handler(request)

    return force_final_answer


def create_limited_agent_middleware(limits: AgentExecutionLimits, extra_middleware: list | None = None):
    middleware = [
        ModelCallLimitMiddleware(
            run_limit=limits.model_calls,
            exit_behavior=limits.model_limit_behavior,
        ),
        _force_final_answer(limits.model_calls),
        ToolCallLimitMiddleware(
            run_limit=limits.tool_calls,
            exit_behavior=limits.tool_limit_behavior,
        ),
    ]

    if limits.tool_retries > 0:
        middleware.append(ToolRetryMiddleware(max_retries=limits.tool_retries))

    return [*middleware, *(extra_middleware or [])]
