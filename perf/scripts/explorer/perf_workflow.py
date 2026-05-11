import time
from typing import Any, List, Optional, cast

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows.workflow import Task, Workflow


class PerfWorkflow(Workflow):
    """A workflow for performance testing of Explorer with OpenAI API calls."""

    is_async: bool = True
    can_reset: bool = True

    def __init__(
        self,
        *,
        task: Task,
        model: ModelWrapper,
        auxiliary_models: Optional[List[ModelWrapper]] = None,
    ):
        super().__init__(
            task=task,
            model=model,
            auxiliary_models=auxiliary_models,
        )
        self.client = self.model.get_openai_async_client()
        self.model_path = getattr(self.client, "model_path")
        self.reset(task)

    def reset(self, task: Task) -> None:
        raw_task = task.raw_task or {}
        self.messages = raw_task.get("messages") or []
        if not self.messages:
            raise ValueError("PerfWorkflow requires task.raw_task['messages'].")
        self.tools = raw_task.get("tools")

    async def run_async(self) -> List[Experience]:
        request_latencies = []
        usage_prompt_tokens = 0.0
        usage_completion_tokens = 0.0
        for i in range(len(self.messages)):
            if self.messages[i].get("role") == "assistant":
                # send a fake request to trigger the workflow and measure performance, but ignore the response content
                request_kwargs = {
                    "model": self.model_path,
                    "messages": self.messages[:i],
                }
                if self.tools is not None:
                    request_kwargs["tools"] = self.tools

                request_start = time.perf_counter()
                responses = await self.client.chat.completions.create(**request_kwargs)
                request_latency = time.perf_counter() - request_start
                request_latencies.append(request_latency)

                usage = cast(Any, getattr(responses, "usage", None))
                prompt_tokens = getattr(usage, "prompt_tokens", None)
                completion_tokens = getattr(usage, "completion_tokens", None)
                if isinstance(prompt_tokens, (int, float)):
                    usage_prompt_tokens += float(prompt_tokens)
                if isinstance(completion_tokens, (int, float)):
                    usage_completion_tokens += float(completion_tokens)

                self.logger.info("Received response: %s", responses.choices[0].message)
        exps = self.model.extract_experience_from_history()
        total_request_latency = sum(request_latencies)
        exps[0].metrics = {
            "prompt_length": usage_prompt_tokens,
            "response_length": usage_completion_tokens,
            "api_call_prompt_tokens_per_second": (
                usage_prompt_tokens / total_request_latency if total_request_latency > 0 else 0.0
            ),
            "api_call_response_tokens_per_second": (
                usage_completion_tokens / total_request_latency
                if total_request_latency > 0
                else 0.0
            ),
        }
        return exps
