import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from examples.learn_to_ask.data_prepare.message_splitter import split_session_to_json_lines
from examples.research_cod.exp_plan_learn2ask.data_prepare.llm_info_extraction import (
    LLM_info_extraction_batch,
    parse_llm_output,
)


def process_jsonl_file(
    input_file, output_file, model_call_mode="online_api", max_retries=3, **kwargs
):
    """Batched info extraction: collect every cid's prompt from every
    session, feed the whole lot to the backend in one llm.generate call
    (for local_vllm) so vLLM's continuous batching can do the work.

    Failed cids are retried in additional batched rounds; at most
    max_retries + 1 rounds total. Surviving failures get info_set=None
    (2_build_dataset drops those with decision=continue).

    Args:
        input_file (str): path to input jsonl (one session per line).
        output_file (str): path to output jsonl (one cid per line, with
            `info_set` filled in).
        model_call_mode (str): "online_api" or "local_vllm".
        max_retries (int): extra rounds of batched retry on top of the
            first attempt.
        **kwargs: forwarded to the backend (model_path, enable_thinking,
            tensor_parallel_size, etc.).
    """
    # -- Load all sessions --
    sessions = []
    with open(input_file, "r", encoding="utf-8") as infile:
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
            try:
                sessions.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON at line {line_num}: {e}")
    print(f"[load] {len(sessions)} sessions from {input_file}")

    # -- Split every session into cids, keep flat + nested views --
    per_session_cids = []  # list[list[dict]] preserving session order
    flat_data = []         # flat list of cid dicts (same objects as nested)
    remaining_chats = []   # flat list of remaining_chat strings

    for session in sessions:
        session_cids = []
        for cid_json in split_session_to_json_lines(session):
            data = json.loads(cid_json)
            data.setdefault("info_set", None)
            session_cids.append(data)
            flat_data.append(data)
            remaining_chats.append(data.get("remaining_chat", ""))
        per_session_cids.append(session_cids)

    total = len(flat_data)
    print(f"[flatten] {total} cids across {len(sessions)} sessions")
    if total == 0:
        # Still emit an empty output so downstream doesn't trip over a missing file.
        open(output_file, "w", encoding="utf-8").close()
        return f"Nothing to process; wrote empty file {output_file}"

    # -- Batched generation with multi-round retry --
    pending = list(range(total))
    for round_idx in range(max_retries + 1):
        if not pending:
            break
        print(
            f"[batch] round {round_idx + 1}/{max_retries + 1}: "
            f"generating for {len(pending)} cids"
        )
        chats_to_run = [remaining_chats[i] for i in pending]
        outputs = LLM_info_extraction_batch(
            chats_to_run, model_call_mode, **kwargs
        )

        still_pending = []
        for orig_idx, text in zip(pending, outputs):
            parsed = parse_llm_output(text)
            if isinstance(parsed, list):
                flat_data[orig_idx]["info_set"] = parsed
            else:
                still_pending.append(orig_idx)
        print(
            f"[batch] round {round_idx + 1}: "
            f"{len(pending) - len(still_pending)} succeeded, "
            f"{len(still_pending)} still pending"
        )
        pending = still_pending

    if pending:
        print(
            f"[batch] WARNING: {len(pending)} cids still failed after "
            f"{max_retries + 1} rounds; leaving info_set=None"
        )

    # -- Write output in session-major order (session 0's cids, then 1's, ...) --
    with open(output_file, "w", encoding="utf-8") as outf:
        for session_cids in per_session_cids:
            for data in session_cids:
                outf.write(json.dumps(data, ensure_ascii=False) + "\n")

    return f"Successfully processed {total} cids. Results saved to {output_file}"


# Example usage:
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file", type=str, default="examples/learn_to_ask/data_raw/train_origin.jsonl"
    )
    parser.add_argument(
        "--output_file", type=str, default="examples/learn_to_ask/data_raw/train_processed.jsonl"
    )
    parser.add_argument(
        "--model_call_mode", type=str, choices=["online_api", "local_vllm"], default="local_vllm"
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--enable_thinking",
        choices=["true", "false"],
        default=None,
        help=(
            "For Qwen3-family local_vllm runs, explicitly toggle the chat "
            "template's thinking mode. Omit to use the model default (which "
            "is True for Qwen3, producing <think>...</think> output that "
            "breaks ast.literal_eval downstream — pass 'false' for extraction)."
        ),
    )
    parser.add_argument(
        "--dataset_id",
        type=str,
        default=None,
        help=(
            "Dataset id on HF or ModelScope (e.g. 'JasonHaggard/RealMedConv'). When set, "
            "--input_file is treated as a *destination path* for dumping the "
            "HF split to jsonl before extraction, instead of an existing file."
        ),
    )
    parser.add_argument(
        "--hf_split",
        type=str,
        default=None,
        help="HF split name (required with --dataset_id), e.g. 'train' / 'test'. "
        "Use --hf_splits instead to process several splits with a single engine.",
    )
    parser.add_argument(
        "--hf_splits",
        nargs="+",
        default=None,
        help=(
            "Multiple HF split names (e.g. 'train test') to process in ONE "
            "process so vLLM engine is initialised only once. Overrides "
            "--hf_split. When --dataset_id is given but neither --hf_split "
            "nor --hf_splits is set, defaults to ['train', 'test']."
        ),
    )
    parser.add_argument(
        "--output_template",
        type=str,
        default="examples/research_cod/data/learn2ask_artifacts/{split}_processed.jsonl",
        help=(
            "Output path template used in multi-split mode; must contain the "
            "literal '{split}'. Default: "
            "examples/research_cod/data/learn2ask_artifacts/{split}_processed.jsonl. "
            "Kept out of examples/research_cod/data/learn2ask/ because HF "
            "datasets 4.x treats that dir as the training base_path and its "
            "recursive ** glob would sweep every train_*/test_*.jsonl into "
            "the respective split."
        ),
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Extra rounds of batched retry on top of the first attempt (default: 3).",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="GPUs to shard one model across (vLLM tensor parallelism). For "
        "single-process offline use, set to your GPU count (e.g. 8) to use all "
        "cards; default: 1.",
    )
    parser.add_argument(
        "--data_parallel_size",
        type=int,
        default=1,
        help="vLLM data parallelism (independent replicas). The single-process "
        "offline LLM only supports 1; >1 raises ValueError and needs the "
        "multi-process launcher. Default: 1.",
    )
    parser.add_argument(
        "--dataset_source",
        choices=["hf", "modelscope"],
        default="modelscope",
        help="Download --dataset_id from HuggingFace or ModelScope (default: modelscope).",
    )
    args = parser.parse_args()

    extra_kwargs = {
        "tensor_parallel_size": args.tensor_parallel_size,
        "data_parallel_size": args.data_parallel_size,
    }
    if args.enable_thinking is not None:
        extra_kwargs["enable_thinking"] = args.enable_thinking == "true"

    # Default to ['train', 'test'] in one shot when the caller gave
    # --dataset_id but did not pin down a single split.
    if (
        args.dataset_id is not None
        and args.hf_splits is None
        and args.hf_split is None
    ):
        args.hf_splits = ["train", "test"]

    # -- Multi-split HF mode: one engine for all splits. --
    # Inputs (HF split dumps) go next to the outputs so the data prep
    # directory holds everything.
    if args.hf_splits:
        assert args.dataset_id is not None, "--hf_splits requires --dataset_id"
        assert (
            args.output_template and "{split}" in args.output_template
        ), "--hf_splits requires --output_template containing '{split}'"
        for split in args.hf_splits:
            output_path = args.output_template.replace("{split}", split)
            input_path = os.path.join(
                os.path.dirname(os.path.abspath(output_path)),
                f"{split}_raw.jsonl",
            )
            os.makedirs(os.path.dirname(os.path.abspath(input_path)), exist_ok=True)
            if args.dataset_source == "modelscope":
                from modelscope.msdatasets import MsDataset  # lazy import
                ds = MsDataset.load(args.dataset_id, split=split)
                n_rows = 0
                with open(input_path, "w", encoding="utf-8") as wf:
                    for item in ds:
                        wf.write(json.dumps(dict(item), ensure_ascii=False) + "\n")
                        n_rows += 1
            else:
                from datasets import load_dataset  # lazy import
                ds = load_dataset(args.dataset_id, split=split)
                ds.to_json(input_path, lines=True, force_ascii=False)
                n_rows = len(ds)
            print(f"[{args.dataset_source}] split={split}: {n_rows} rows -> {input_path}")
            print(
                process_jsonl_file(
                    input_file=input_path,
                    output_file=output_path,
                    model_call_mode=args.model_call_mode,
                    model_path=args.model_path,
                    max_retries=args.max_retries,
                    **extra_kwargs,
                )
            )
    else:
        # -- Single-split path (backward compatible). --
        if args.dataset_id is not None:
            assert args.hf_split is not None, "--hf_split is required with --dataset_id"
            from datasets import load_dataset  # lazy import
            ds = load_dataset(args.dataset_id, split=args.hf_split)
            os.makedirs(os.path.dirname(os.path.abspath(args.input_file)), exist_ok=True)
            ds.to_json(args.input_file, lines=True, force_ascii=False)
            print(
                f"[hf] loaded {len(ds)} rows from {args.dataset_id}:{args.hf_split} "
                f"and dumped to {args.input_file}"
            )

        print(
            process_jsonl_file(
                input_file=args.input_file,
                output_file=args.output_file,
                model_call_mode=args.model_call_mode,
                model_path=args.model_path,
                max_retries=args.max_retries,
                **extra_kwargs,
            )
        )
