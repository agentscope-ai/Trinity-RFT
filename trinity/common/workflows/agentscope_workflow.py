import inspect
from typing import Awaitable, Callable, Dict, List, Optional

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows.workflow import Task, Workflow
from trinity.utils.annotations import Deprecated


@Deprecated
class AgentScopeWorkflowAdapter(Workflow):
    """Adapter to wrap a agentscope trainable workflow function into a Trinity Workflow.
    Only for agentscope versions between 1.0.7 and 1.0.11.
    For agentscope >= 1.0.12, please use AgentScopeWorkflowAdapterV1.
    """

    is_async: bool = True

    def __init__(
        self,
        *,
        task: Task,
        model: ModelWrapper,
        auxiliary_models: Optional[List[ModelWrapper]] = None,
    ):
        """Initialize the adapter with the task and model."""
        try:
            from agentscope.model import TrinityChatModel
        except ImportError:
            raise ImportError(
                "This workflow requires agentscope >= 1.0.7, please install "
                "it via `pip install agentscope>=1.0.7`",
            )

        super().__init__(
            task=task,
            model=model,
            auxiliary_models=auxiliary_models,
        )
        self.workflow_func: Callable[
            [Dict, TrinityChatModel], Awaitable[float]
        ] = task.workflow_args.get("workflow_func", None)

        if self.workflow_func is None:
            raise ValueError(
                "The 'workflow_func' is not provided.",
            )

        self.chat_model: TrinityChatModel = TrinityChatModel(
            model.get_openai_async_client(),
            generate_kwargs={
                "temperature": self.task.rollout_args.temperature,
                "top_p": self.task.rollout_args.top_p,
                "max_tokens": self.task.rollout_args.max_tokens or 4096,
                "logprobs": True,
                "top_logprobs": self.task.rollout_args.logprobs,
            },
        )

    def construct_experiences(
        self,
        reward: float,
    ) -> List[Experience]:
        """Construct experiences from the agent's interaction history.

        Args:
            reward (float): The reward value to assign to each experience.

        Returns:
            List: A list of Experience objects.
        """
        exps = self.model.extract_experience_from_history()
        for exp in exps:
            exp.reward = reward
        return exps

    async def run_async(self) -> List[Experience]:
        """Run the workflow asynchronously and return experiences."""
        reward = await self.workflow_func(self.task.raw_task, self.chat_model)  # type: ignore [arg-type]
        return self.construct_experiences(reward)


class AgentScopeWorkflowAdapterV1(Workflow):
    """A more general adapter to wrap agentscope trainable workflow and judge functions into a Trinity Workflow.
    Only for agentscope versions >= 1.0.12.
    """

    is_async: bool = True

    def __init__(
        self,
        *,
        task: Task,
        model: ModelWrapper,
        auxiliary_models: Optional[List[ModelWrapper]] = None,
    ):
        """Initialize the adapter with the task and model."""
        try:
            from agentscope.model import OpenAIChatModel
        except ImportError:
            raise ImportError(
                "This workflow requires agentscope >= 1.0.12, please install "
                "it via `pip install agentscope>=1.0.12`",
            )

        super().__init__(
            task=task,
            model=model,
            auxiliary_models=auxiliary_models,
        )
        self.workflow_func = task.workflow_args.get("workflow_func", None)
        self.judge_func = task.workflow_args.get("judge_func", None)
        self._openai_client = self.model.get_openai_async_client()

        if self.workflow_func is None:
            raise ValueError(
                "The 'workflow_func' is not provided.",
            )

        self.chat_model: OpenAIChatModel = OpenAIChatModel(
            api_key="EMPTY",
            model_name=self._openai_client.model_path,
            stream=False,
            generate_kwargs={
                "temperature": self.task.rollout_args.temperature,
                "top_p": self.task.rollout_args.top_p,
                "max_tokens": self.task.rollout_args.max_tokens or 4096,
                "logprobs": True,
                "top_logprobs": self.task.rollout_args.logprobs,
            },
        )
        self.chat_model.client = self._openai_client
        self.auxiliary_chat_models: Dict[str, OpenAIChatModel] = {}
        if self.auxiliary_model_wrappers is not None:
            for aux_model_wrapper in self.auxiliary_model_wrappers:
                aux_model_client = aux_model_wrapper.get_openai_async_client()
                aux_chat_model = OpenAIChatModel(
                    api_key="EMPTY",
                    model_name=aux_model_client.model_path,
                    generate_kwargs=aux_model_wrapper.generate_kwargs,
                    stream=False,
                )
                aux_chat_model.client = aux_model_client
                assert (
                    aux_model_wrapper.model_name is not None
                ), "Auxiliary model must have a name. This should not happen."
                self.auxiliary_chat_models[aux_model_wrapper.model_name] = aux_chat_model

    def construct_experiences(
        self,
        reward: float,
        metrics: Dict,
    ) -> List[Experience]:
        """Construct experiences from the agent's interaction history.

        Args:
            reward (float): The reward value to assign to each experience.
            metrics (Dict): A dictionary of metrics to be attached to the last experience.

        Returns:
            List: A list of Experience objects.
        """
        exps = self.model.extract_experience_from_history()
        for exp in exps:
            exp.reward = reward
        # only attach metrics to the last experience
        if len(exps) > 0:
            exps[-1].metrics = metrics
        return exps

    async def run_async(self) -> List[Experience]:
        """Run the workflow asynchronously and return experiences."""
        try:
            from agentscope.tuner import JudgeOutput, WorkflowOutput
        except ImportError:
            raise ImportError(
                "Fail to import agentscope tuner related types. Please ensure agentscope>=1.0.12 is installed."
            )

        metrics = {}
        workflow_sig = inspect.signature(self.workflow_func)
        workflow_input_dict = {
            "task": self.task.raw_task,
            "model": self.chat_model,
        }
        if "auxiliary_models" in workflow_sig.parameters:
            workflow_input_dict["auxiliary_models"] = self.auxiliary_chat_models
        if "logger" in workflow_sig.parameters:
            workflow_input_dict["logger"] = self.logger
        workflow_output = await self.workflow_func(**workflow_input_dict)
        if not isinstance(workflow_output, WorkflowOutput):
            raise ValueError(
                "The 'workflow_func' must return a WorkflowOutput object.",
            )
        metrics.update(workflow_output.metrics or {})
        if self.judge_func is not None:
            judge_sig = inspect.signature(self.judge_func)
            judge_input_dict = {
                "task": self.task.raw_task,
                "response": workflow_output.response,
            }
            if "auxiliary_models" in judge_sig.parameters:
                judge_input_dict["auxiliary_models"] = self.auxiliary_chat_models
            if "logger" in judge_sig.parameters:
                judge_input_dict["logger"] = self.logger
            judge_output = await self.judge_func(**judge_input_dict)
            if not isinstance(judge_output, JudgeOutput):
                raise ValueError(
                    "The 'judge_func' must return a JudgeOutput object.",
                )
            reward = judge_output.reward
            metrics.update(judge_output.metrics or {})
        else:
            assert (
                workflow_output.reward is not None
            ), "Either workflow or judge must provide reward."
            reward = workflow_output.reward
        return self.construct_experiences(reward, metrics)
