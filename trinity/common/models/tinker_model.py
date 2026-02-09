import time
from os import getenv
from typing import List, Optional, Sequence

import ray
import tinker
import torch
from tinker import types
from torch import Tensor

from trinity.common.config import InferenceModelConfig
from trinity.common.experience import Experience
from trinity.common.models.model import BaseInferenceModel
from trinity.manager.synchronizer import Synchronizer


class TinkerModel(BaseInferenceModel):
    def __init__(
        self,
        config: InferenceModelConfig,
    ) -> None:
        super().__init__(config)
        self.model_version = -1
        self.synchronizer = Synchronizer.get_actor(namespace=ray.get_runtime_context().namespace)
        self.model = None
        self.model_path = config.model_path

    async def _initialize_tokenizer(self) -> None:
        """Initialize the tokenizer."""
        self.tokenizer = self.model.get_tokenizer()

    async def _generate_internal(self, prompt: dict, **kwargs) -> types.SampleResponse:
        assert self.model is not None
        sampling_params = {
            "max_tokens": kwargs.get("max_tokens", self.config.max_response_tokens),
            "seed": kwargs.get("seed", self.config.seed),
            "temperature": kwargs.get("temperature", 1.0),
            "top_k": kwargs.get("top_k", -1),
            "top_p": kwargs.get("top_p", 1),
        }

        return await self.model.sample_async(
            prompt=types.ModelInput.from_ints(prompt["prompt_token_ids"]),
            sampling_params=sampling_params,
            num_samples=kwargs.get("n", 1),
            include_prompt_logprobs=kwargs.get("include_prompt_logprobs", False),
            topk_prompt_logprobs=kwargs.get("topk_prompt_logprobs", self.config.logprobs),
        )

    async def generate(self, prompt: str, **kwargs) -> Sequence[Experience]:
        """Generate a responses from a prompt in async."""
        if self.tokenizer is None:
            await self._initialize_tokenizer()

        token_ids, is_valid = self._handle_prompt_truncation(prompt, **kwargs)
        if not is_valid:
            return token_ids

        with_chat_completion = kwargs.get("with_chat_completion", False)
        if with_chat_completion:
            create_time = int(time.time())
        output = await self._generate_internal(prompt={"prompt_token_ids": token_ids}, **kwargs)
        logprobs = kwargs.get("logprobs", self.config.logprobs)
        return_logprobs = logprobs is not None and logprobs is not False
        experiences = [
            Experience(
                tokens=torch.tensor(token_ids + sequence.tokens, dtype=torch.int32),
                logprobs=(
                    torch.tensor(sequence.logprobs, dtype=torch.float32)
                    if return_logprobs
                    else torch.tensor([], dtype=torch.float32)
                ),
                prompt_length=len(token_ids),
                prompt_text=self.tokenizer.decode(token_ids),
                response_text=self.tokenizer.decode(sequence.tokens),
            )
            for sequence in output.sequences
        ]
        if with_chat_completion:
            from openai.types.chat.chat_completion import (
                ChatCompletion,
                ChatCompletionMessage,
                ChatCompletionTokenLogprob,
                Choice,
                ChoiceLogprobs,
            )

            return_token_ids = kwargs.get("return_token_ids", False)
            chat_completion = ChatCompletion(
                id="",
                choices=[
                    Choice(
                        finish_reason=sequence.stop_reason,
                        index=i,
                        logprobs=ChoiceLogprobs(
                            content=[
                                ChatCompletionTokenLogprob(
                                    token=self.tokenizer.decode(token_id),
                                    logprob=logprob,
                                    top_logprobs=[],
                                )
                                for token_id, logprob in zip(sequence.tokens, sequence.logprobs)
                            ]
                        ),
                        message=ChatCompletionMessage(
                            content=self.tokenizer.decode(sequence.tokens), role="assistant"
                        ),
                        token_ids=(sequence.tokens if return_token_ids else None),
                    )
                    for i, sequence in enumerate(output.sequences)
                ],
                created=create_time,
                model=self.model_path,
                object="chat.completion",
                prompt_token_ids=token_ids,
            )
            experiences.append(chat_completion)

        return experiences

    async def chat(self, messages: List[dict], **kwargs) -> Sequence[Experience]:
        """Generate experiences from a list of history chat messages in async."""
        if self.tokenizer is None:
            await self._initialize_tokenizer()

        # TODO: this is a hack to support openai chat messages, which only supports text
        for msg in messages:
            if isinstance(msg["content"], list):
                text_parts = [item["text"] for item in msg["content"] if item["type"] == "text"]
                content_str = "".join(text_parts)
            else:
                content_str = msg["content"]
            msg["content"] = content_str

        prompt = self.apply_chat_template(self.tokenizer, messages)
        return await self.generate(prompt=prompt, **kwargs)

    async def logprobs(self, token_ids: List[int], **kwargs) -> Tensor:
        """Generate logprobs for a list of tokens in async."""
        logprobs = await self.model.compute_logprobs_async(types.ModelInput.from_ints(token_ids))
        return torch.tensor(logprobs[1:], dtype=torch.float32)

    async def prepare(self) -> None:
        """Prepare the model before inference."""
        self.service_client = tinker.ServiceClient()
        self.model = await self.service_client.create_sampling_client_async(
            base_model=self.config.model_path,
        )
        await self._initialize_tokenizer()

    async def sync_model(self, model_version: int) -> int:
        self.model_version = model_version
        remote_sampler_path, _ = await self.synchronizer.get_model_state_dict.remote()
        self.model = await self.service_client.create_sampling_client_async(
            model_path=remote_sampler_path,
        )
        self.model_path = remote_sampler_path
        return model_version

    def get_model_version(self) -> int:
        """Get the checkpoint version."""
        return self.model_version

    def get_api_server_url(self) -> Optional[str]:
        """
        Get the Tinker Openai API interface URL.

        Documentation: https://tinker-docs.thinkingmachines.ai/compatible-apis/openai

        Note: This URL is currently not in active use because Tinker's OpenAI-compatible
        API implementation is still incomplete. Instead, we're using our custom `self.chat()`
        method to replicate the functionality of `openai.OpenAI.chat.completions.create()`.

        Once Tinker's API is fully implemented and stable, we plan to switch to using this
        official endpoint directly.
        """
        return "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/"

    def get_api_key(self):
        return getenv("TINKER_API_KEY")

    def get_model_path(self) -> Optional[str]:
        """Get the latest sampler weight path."""
        return self.model_path
