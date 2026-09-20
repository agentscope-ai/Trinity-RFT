"""AgentScope adapters used by CoD workflows."""

import json
from typing import Any, List, Optional

try:
    from agentscope.agent import AgentBase, ReActAgent
    from agentscope.formatter import OpenAIChatFormatter
    from agentscope.memory import InMemoryMemory
    from agentscope.message import Msg
    from agentscope.model import OpenAIChatModel
    from agentscope.tool import Toolkit
except Exception as e:
    _AGENTSCOPE_IMPORT_ERROR = e
else:
    _AGENTSCOPE_IMPORT_ERROR = None


def _ensure_agentscope() -> None:
    if _AGENTSCOPE_IMPORT_ERROR is not None:
        raise ImportError(
            "AgentScope is not installed or failed to import. Please install it "
            "before running CoD AgentScope workflows."
        ) from _AGENTSCOPE_IMPORT_ERROR


def _build_model_generate_kwargs(model) -> dict:
    """Build AgentScope generate kwargs from rollout model config."""
    config = model.config
    kwargs = {
        "temperature": config.temperature,
        "top_p": config.top_p,
        "max_tokens": config.max_response_tokens,
    }
    extra_body = {}
    if config.top_k is not None:
        extra_body["top_k"] = config.top_k
    if config.min_response_tokens is not None:
        extra_body["min_tokens"] = config.min_response_tokens
    if config.repetition_penalty is not None:
        extra_body["repetition_penalty"] = config.repetition_penalty
    if extra_body:
        kwargs["extra_body"] = extra_body
    return kwargs


def _build_rollout_generate_kwargs(rollout_args) -> dict:
    """Build AgentScope generate kwargs from task rollout_args."""
    kwargs = {
        "temperature": rollout_args.temperature,
        "top_p": rollout_args.top_p,
    }
    if rollout_args.max_tokens is not None:
        kwargs["max_tokens"] = rollout_args.max_tokens
    if rollout_args.top_k != -1:
        kwargs["extra_body"] = {"top_k": rollout_args.top_k}
    return kwargs


def _build_agentscope_chat_model(model_path: str, client, generate_kwargs: dict):
    """Build an AgentScope OpenAIChatModel with Trinity's recording client."""
    _ensure_agentscope()

    chat_model = OpenAIChatModel(
        api_key="EMPTY",
        model_name=model_path,
        stream=False,
        generate_kwargs=generate_kwargs,
    )
    chat_model.client = client
    return chat_model


async def _build_agentscope_chat_model_from_trinity(model, generate_kwargs: dict):
    """Build an AgentScope chat model from a Trinity ModelWrapper."""
    model_path = model.model_path
    client = model.get_openai_async_client()
    return _build_agentscope_chat_model(
        model_path=model_path,
        client=client,
        generate_kwargs=generate_kwargs,
    )


async def build_agentscope_single_turn_agent(*, name: str, model, rollout_args):
    """Build an AgentScope agent that maps messages to one model call."""
    _ensure_agentscope()

    class _SingleTurnAgent(AgentBase):
        """Single-turn agent: messages -> one model call -> assistant Msg."""

        def __init__(self, model, formatter):
            super().__init__()
            self.name = name
            self.model = model
            self.formatter = formatter

        async def reply(self, messages: List[dict]) -> "Msg":
            msgs = [Msg(m["role"], m["content"], m["role"]) for m in messages]
            res = await self.model(await self.formatter.format(msgs=msgs))
            return Msg(self.name, res.content, "assistant")

        async def observe(self, msg=None) -> None:
            pass

        async def handle_interrupt(self, *args, **kwargs) -> "Msg":
            return Msg(self.name, "Interrupted.", "assistant")

    generate_kwargs = _build_rollout_generate_kwargs(rollout_args)
    chat_model = await _build_agentscope_chat_model_from_trinity(model, generate_kwargs)
    return _SingleTurnAgent(chat_model, OpenAIChatFormatter())


async def build_agentscope_react_agent(
    *,
    name: str,
    model,
    system_prompt: str,
    compress_assistant_fn,
    toolkit=None,
    max_iters: int = 1,
):
    """Build an AgentScope ReActAgent with compressed assistant text memory."""
    _ensure_agentscope()

    class _CompressingMemory(InMemoryMemory):
        """Compress text-only assistant messages without touching tool blocks."""

        async def add(self, memories, marks=None, allow_duplicates=False, **kwargs):
            if memories is not None:
                msgs = memories if isinstance(memories, list) else [memories]
                out: List[Optional[Msg]] = []
                for msg in msgs:
                    has_tool_blocks = (
                        msg is not None
                        and (
                            msg.has_content_blocks("tool_use")
                            or msg.has_content_blocks("tool_result")
                        )
                    )
                    if (
                        msg is not None
                        and msg.role == "assistant"
                        and not has_tool_blocks
                    ):
                        text = msg.get_text_content() or ""
                        msg = Msg(
                            msg.name,
                            compress_assistant_fn(text),
                            "assistant",
                        )
                    out.append(msg)
                memories = out if isinstance(memories, list) else out[0]
            await super().add(
                memories, marks=marks, allow_duplicates=allow_duplicates, **kwargs
            )

    generate_kwargs = _build_model_generate_kwargs(model)
    chat_model = await _build_agentscope_chat_model_from_trinity(model, generate_kwargs)
    return ReActAgent(
        name=name,
        sys_prompt=system_prompt,
        model=chat_model,
        formatter=OpenAIChatFormatter(),
        toolkit=toolkit if toolkit is not None else Toolkit(),
        memory=_CompressingMemory(),
        max_iters=max_iters,
    )


async def run_agentscope_agent_step(agent, user_content: str) -> str:
    """Run one AgentScope reply and return final text."""
    _ensure_agentscope()

    reply = await agent.reply(Msg("user", user_content, role="user"))
    return reply.get_text_content() or ""


def _format_tool_output(output: Any) -> str:
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        parts = []
        for item in output:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(json.dumps(item, ensure_ascii=False))
        return "\n".join(parts)
    return json.dumps(output, ensure_ascii=False)


def agentscope_msg_to_text(msg) -> str:
    """Render one AgentScope message for CoD logs and trajectories."""
    parts = []
    for block in msg.get_content_blocks():
        block_type = block.get("type")
        if block_type == "text":
            parts.append(block.get("text", ""))
        elif block_type == "tool_use":
            tool_input = json.dumps(block.get("input", {}), ensure_ascii=False)
            parts.append(f"[tool_call] {block.get('name')}({tool_input})")
        elif block_type == "tool_result":
            output = _format_tool_output(block.get("output", ""))
            parts.append(f"[tool_result] {block.get('name')}: {output}")
        elif block_type == "thinking":
            parts.append(f"[thinking] {block.get('thinking', '')}")
        else:
            parts.append(f"[{block_type}]")
    content = "\n".join(part for part in parts if part)
    if not content:
        return ""
    label = "Tool" if msg.has_content_blocks("tool_result") else msg.role.capitalize()
    return f"{label}: {content}"


def agentscope_msgs_to_text(messages: List[Any]) -> str:
    """Render AgentScope messages as a compact transcript."""
    lines = []
    for msg in messages:
        rendered = agentscope_msg_to_text(msg)
        if rendered.strip():
            lines.append(rendered)
    return "\n".join(lines)
