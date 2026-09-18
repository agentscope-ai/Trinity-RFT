import ast
import os
import re
import traceback
from typing import List

import openai

llm = None


# System prompt shared by the single-shot and batched entry points.
_SYSTEM_MESSAGE = """
    # Task:
    You are a medical information assistant. Given a dialogue between a physician (assistant) and a patient (user), extract the clinical attributes of interest to the physician based on their questions. The target fields include: symptom, symptom nature, symptom location, symptom severity, and symptom trigger. Then, identify the corresponding specific information from the patient's responses and pair it with the respective field.
    # Requirements:
        - Do not fabricate information or introduce new fields not listed above. Ignore patient-reported information regarding prior medication use, allergies, or underlying comorbidities; do not include such details in the output.
        - Only include fields explicitly inquired about by the physician. Omit any fields not addressed in the dialogue. Avoid outputting vague terms (e.g., "unspecified" or "unknown").
        - Prevent duplication: if a symptom description already includes anatomical location, do not separately list the location field.
        - Format each entry as a string enclosed in single quotes ('), separate multiple entries with commas, and enclose them in square brackets to form a list. Prefix the list with "output: " on the final line. If the dialogue is unrelated to the aforementioned clinical attributes, output "output: []".
        - Do not include any commentary after the "output:" line. Condense colloquial patient expressions into concise, standardized, and clinically appropriate terminology.
    # Example output format:
    output: ['symptom: diarrhea', 'symptom nature: watery stool', 'symptom severity: 4-5 times per day']
    """


def _build_messages(remaining_chat: str) -> list:
    return [
        {"role": "system", "content": _SYSTEM_MESSAGE},
        {"role": "user", "content": "```\n" + remaining_chat + "\n```\n"},
    ]


def LLM_info_extraction(remaining_chat, model_call_mode, **kwargs):
    """
    Extract information from a single remaining_chat using LLM.

    Kept for backward compatibility and ad-hoc debugging; the main
    pipeline uses LLM_info_extraction_batch for throughput.
    """
    messages = _build_messages(remaining_chat)
    try:
        if model_call_mode == "online_api":
            return _call_online_api(messages, **kwargs)
        elif model_call_mode == "local_vllm":
            return _call_local_vllm(messages, **kwargs)
        else:
            return f"Error: Invalid model_call_mode '{model_call_mode}'. Must be 'online_api' or 'local_vllm'."
    except Exception as e:
        return f"Error occurred: {str(e)}"


def LLM_info_extraction_batch(
    remaining_chats: List[str], model_call_mode: str, **kwargs
) -> List[str]:
    """Batched counterpart of LLM_info_extraction.

    Builds the prompt for every remaining_chat, then issues a single
    llm.generate(all_prompts) call so vLLM's continuous batching can
    fully utilise the GPUs. Returns a list of response texts, one per
    input chat, in the same order.

    online_api mode has no true batching (DashScope etc. are per-call),
    so it just falls back to a sequential loop.
    """
    messages_list = [_build_messages(rc) for rc in remaining_chats]

    if model_call_mode == "local_vllm":
        return _call_local_vllm_batch(messages_list, **kwargs)
    if model_call_mode == "online_api":
        return [_call_online_api(m, **kwargs) for m in messages_list]
    err = f"Error: Invalid model_call_mode '{model_call_mode}'. Must be 'online_api' or 'local_vllm'."
    return [err] * len(remaining_chats)


def _call_online_api(messages, **kwargs):
    """Handle OpenAI-style API calls"""
    # Extract API parameters from kwargs or use defaults
    api_key = kwargs.get("api_key", os.getenv("DASHSCOPE_API_KEY"))
    api_base = kwargs.get("api_base", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    model = kwargs.get("model", "qwen2.5-72b-instruct")
    temperature = kwargs.get("temperature", 0.7)
    max_tokens = kwargs.get("max_tokens", 500)

    client = openai.OpenAI(api_key=api_key, base_url=api_base)
    response = client.chat.completions.create(
        model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
    )

    return response.choices[0].message.content


def _vllm_setup(**kwargs):
    """Lazy-init (and cache) the vLLM engine. Returns the global llm and
    the SamplingParams to use."""
    from vllm import LLM, SamplingParams

    model_path = kwargs.get("model_path")
    if not model_path:
        raise ValueError("model_path is required for local vLLM inference")

    # Qwen3.6 thinking general, official: temp 1.0 / top_p 0.95 / top_k 20 / min_p 0 /
    # presence_penalty 0. max_tokens 32768 per the card — <think> needs room before "output:".
    temperature = 1.0
    top_p = 0.95
    top_k = 20
    min_p = 0.0
    presence_penalty = 0.0
    max_tokens = 32768

    tensor_parallel_size = kwargs.get("tensor_parallel_size", 1)
    data_parallel_size = kwargs.get("data_parallel_size", 1)
    gpu_memory_utilization = kwargs.get("gpu_memory_utilization", 0.9)
    enforce_eager = kwargs.get("enforce_eager", False)
    dtype = kwargs.get("dtype", "auto")
    max_model_len = 40960  # 32768 output + headroom for the dialogue prompt

    global llm
    if llm is None:
        llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            data_parallel_size=data_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            dtype=dtype,
            max_model_len=max_model_len,
        )

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        presence_penalty=presence_penalty,
        max_tokens=max_tokens,
    )
    return llm, sampling_params


def _call_local_vllm(messages, **kwargs):
    """Single-conversation local vLLM call via llm.chat: the engine renders
    the chat template, and enable_thinking is passed via chat_template_kwargs."""
    try:
        llm, sampling_params = _vllm_setup(**kwargs)
        et = kwargs.get("enable_thinking")
        ctk = {"enable_thinking": et} if et is not None else None
        outputs = llm.chat(messages, sampling_params, chat_template_kwargs=ctk)
        return outputs[0].outputs[0].text
    except ImportError:
        return "Error: vLLM library not installed. Please install it with 'pip install vllm'"
    except Exception as e:
        traceback.print_exc()  # surface the real error instead of silently swallowing it
        return f"Error in local vLLM inference: {str(e)}"


def _call_local_vllm_batch(messages_list: List[list], **kwargs) -> List[str]:
    """Batched local vLLM call: hand all conversations to one llm.chat so
    vLLM continuous-batches internally. Returns one output text per input,
    in order. enable_thinking is passed via chat_template_kwargs.
    """
    if not messages_list:
        return []
    try:
        llm, sampling_params = _vllm_setup(**kwargs)
        et = kwargs.get("enable_thinking")
        ctk = {"enable_thinking": et} if et is not None else None
        outputs = llm.chat(messages_list, sampling_params, chat_template_kwargs=ctk)
        return [o.outputs[0].text for o in outputs]
    except ImportError:
        return ["Error: vLLM library not installed. Please install it with 'pip install vllm'"] * len(messages_list)
    except Exception as e:
        traceback.print_exc()  # surface the real error instead of silently swallowing it
        return [f"Error in local vLLM inference: {str(e)}"] * len(messages_list)


def parse_llm_output(output_str):
    """
    Convert the LLM info extraction output string to a list of strings.

    Args:
        output_str (str): model output ending with "output: [...]" (a bracketed
            list of "field: value" strings; "output: []" if nothing relevant).

    Returns:
        list: List of strings if successful, error message string if failed
    """
    # Lock onto the "output: [...]" format; take the LAST match so any
    # <think> reasoning or prose before it is ignored.
    matches = re.findall(r"output:\s*(\[[^][]*\])", output_str, flags=re.DOTALL | re.IGNORECASE)
    if not matches:
        return f"Error parsing output: no 'output: [...]' found in [{repr(output_str)}]"
    try:
        result = ast.literal_eval(matches[-1])
        if not isinstance(result, list):
            return f"Error: Expected a list, got {type(result)}"
        return result
    except Exception as e:
        return f"Error parsing output: [{repr(matches[-1])}] error = {str(e)}"
