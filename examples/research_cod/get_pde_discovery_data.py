"""Generate CoD PDE discovery tasksets.

The dataset intentionally does not pre-bind rows to CoD packs. Pack-level
randomness is injected at runtime by CoDWorkflow via ``pack_seed`` so changing
pack size or mixing tasksets does not require regenerating the dataset.
"""

import argparse
import importlib.util
import os
import sys
import types
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "pde_discovery"
)


def save_dataset_to_local(data_path: str, data: list[dict], split: str) -> str:
    os.makedirs(data_path, exist_ok=True)
    dataset_path = os.path.join(data_path, f"{split}.parquet")
    pd.DataFrame(data).to_parquet(dataset_path)
    print(f"Saved split '{split}' with {len(data)} examples at {dataset_path}")
    return dataset_path


def prepare_pde_discovery_data(
    data_path: str,
    train_size: int,
    test_size: int,
    seed: int,
    eval_pack_size: int | None = None,
    eval_template_count: int | None = None,
    exclude_eval_paths: list[str] | None = None,
    eval_ground_truth_family: str = "physical_full_eval_hard.json",
    eval_min_reaction_terms: int = 1,
    eval_max_reaction_terms: int = 3,
    eval_state_abs_limit: float = 3.0,
    eval_initial_mode_count_range: tuple[int, int] = (1, 2),
    save_train_split: bool = True,
    online_eval_data_path: str | None = None,
    online_eval_instances_per_template: int | None = None,
) -> tuple[list[dict], list[dict]]:
    counts = {
        "train_size": train_size,
        "test_size": test_size,
    }
    invalid = [name for name, value in counts.items() if value <= 0]
    if invalid:
        raise ValueError(f"PDE dataset counts must be positive: {', '.join(invalid)}")
    if (online_eval_data_path is None) != (online_eval_instances_per_template is None):
        raise ValueError(
            "online_eval_data_path and online_eval_instances_per_template "
            "must be provided together"
        )
    if online_eval_data_path is not None:
        if os.path.abspath(online_eval_data_path) == os.path.abspath(data_path):
            raise ValueError(
                "Online eval directory must differ from the full dataset directory"
            )
        stale_online_train_path = os.path.join(online_eval_data_path, "train.parquet")
        if os.path.exists(stale_online_train_path):
            raise ValueError(
                "Online test-only directory contains train.parquet: "
                f"{stale_online_train_path}"
            )

    rng = np.random.default_rng(seed)
    uint32_space_size = int(np.iinfo(np.uint32).max) + 1
    all_task_seeds = rng.choice(
        uint32_space_size,
        size=train_size + test_size,
        replace=False,
    )

    eval_template_assignments = build_stratified_eval_template_assignments(
        test_size=test_size,
        eval_pack_size=eval_pack_size,
        eval_template_count=eval_template_count,
    )

    def build_split(task_seeds, split_name, template_assignments=None):
        rows = []
        for task_idx, task_seed in enumerate(task_seeds):
            row = {
                "uid": f"pde_{split_name}_{task_idx}",
                "seed": int(task_seed),
                "task_desc": (
                    "Discover the nonlinear reaction term f(u) in "
                    "partial_t u = partial_xx u + f(u) using active "
                    "sampling, sparse regression, and scientific context."
                ),
                "answer": "",
            }
            if template_assignments is not None:
                assignment = template_assignments[task_idx]
                row.update(assignment)
                pack_start = (task_idx // eval_pack_size) * eval_pack_size
                row["pde_environment_seed"] = int(task_seeds[pack_start])
            rows.append(row)
        return rows

    train_data = build_split(
        all_task_seeds[:train_size],
        "train",
    )
    test_data = build_split(
        all_task_seeds[train_size:],
        "test",
        template_assignments=eval_template_assignments,
    )
    if exclude_eval_paths:
        if eval_pack_size is None or eval_template_count is None:
            raise ValueError(
                "Equation-disjoint evaluation requires eval_pack_size and "
                "eval_template_count"
            )
        validate_eval_equation_disjointness(
            new_rows=test_data,
            exclude_eval_paths=exclude_eval_paths,
            eval_pack_size=eval_pack_size,
            ground_truth_family=eval_ground_truth_family,
            min_reaction_terms=eval_min_reaction_terms,
            max_reaction_terms=eval_max_reaction_terms,
            state_abs_limit=eval_state_abs_limit,
            initial_mode_count_range=eval_initial_mode_count_range,
        )

    if save_train_split:
        save_dataset_to_local(data_path, train_data, "train")
    else:
        stale_train_path = os.path.join(data_path, "train.parquet")
        if os.path.exists(stale_train_path):
            raise ValueError(
                f"--test_only requires a directory without train.parquet: {stale_train_path}"
            )
    save_dataset_to_local(data_path, test_data, "test")
    if online_eval_data_path is not None:
        assert online_eval_instances_per_template is not None
        if eval_pack_size is None or eval_template_count is None:
            raise ValueError(
                "Online eval export requires eval_pack_size and eval_template_count"
            )
        online_test_data = select_eval_template_instances(
            test_data,
            eval_pack_size=eval_pack_size,
            eval_template_count=eval_template_count,
            instances_per_template=online_eval_instances_per_template,
        )
        save_dataset_to_local(online_eval_data_path, online_test_data, "test")
    return train_data, test_data


def _resolve_test_parquet(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / "test.parquet"
    if not candidate.is_file():
        raise FileNotFoundError(f"Excluded eval test parquet does not exist: {candidate}")
    return candidate


def _eval_pack_descriptors(rows: Iterable[dict], eval_pack_size: int) -> list[dict]:
    """Return one hidden-environment descriptor for every fixed eval pack."""
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        if "eval_pack_index" not in row:
            raise ValueError("Stratified eval row is missing eval_pack_index")
        grouped.setdefault(int(row["eval_pack_index"]), []).append(row)

    descriptors = []
    for pack_index in sorted(grouped):
        pack = grouped[pack_index]
        if len(pack) != eval_pack_size:
            raise ValueError(
                f"Eval pack {pack_index} has {len(pack)} rows, expected {eval_pack_size}"
            )
        template_indices = {int(row["ground_truth_template_index"]) for row in pack}
        environment_seeds = {int(row["pde_environment_seed"]) for row in pack}
        if len(template_indices) != 1 or len(environment_seeds) != 1:
            raise ValueError(
                f"Eval pack {pack_index} does not share one template and environment seed"
            )
        descriptors.append(
            {
                "pack_index": pack_index,
                "template_index": template_indices.pop(),
                "environment_seed": environment_seeds.pop(),
            }
        )
    return descriptors


def select_eval_template_instances(
    rows: list[dict],
    eval_pack_size: int,
    eval_template_count: int,
    instances_per_template: int,
) -> list[dict]:
    """Select the first fixed equation instances of every eval template."""
    if instances_per_template <= 0:
        raise ValueError("Online eval instances per template must be positive")
    selected = [
        row
        for row in rows
        if int(row["ground_truth_template_instance"]) < instances_per_template
    ]
    descriptors = _eval_pack_descriptors(selected, eval_pack_size)
    template_counts = Counter(descriptor["template_index"] for descriptor in descriptors)
    expected_pack_count = eval_template_count * instances_per_template
    if len(descriptors) != expected_pack_count:
        raise ValueError(
            f"Online eval selected {len(descriptors)} packs, expected {expected_pack_count}"
        )
    if set(template_counts) != set(range(eval_template_count)) or set(
        template_counts.values()
    ) != {instances_per_template}:
        raise ValueError("Online eval does not cover every template equally")
    return selected


def _load_pde_ground_truth_modules():
    """Load equation-sampling modules without importing the vLLM workflow."""
    root = (
        Path(__file__).parents[2]
        / "trinity/common/workflows/connect_the_dots/pde_discovery"
    )
    package_name = "_pde_eval_generation_modules"
    package = sys.modules.setdefault(package_name, types.ModuleType(package_name))
    package.__path__ = [str(root)]

    loaded = {}
    for name in ("candidate", "pde_numeric", "ground_truth"):
        module_name = f"{package_name}.{name}"
        if module_name in sys.modules:
            loaded[name] = sys.modules[module_name]
            continue
        spec = importlib.util.spec_from_file_location(module_name, root / f"{name}.py")
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load PDE equation module: {name}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        loaded[name] = module
    return loaded["candidate"], loaded["ground_truth"], loaded["pde_numeric"]


def _hidden_equation_signatures(
    rows: Iterable[dict],
    eval_pack_size: int,
    ground_truth_family: str,
    min_reaction_terms: int,
    max_reaction_terms: int,
    state_abs_limit: float,
    initial_mode_count_range: tuple[int, int],
) -> list[tuple[tuple[tuple[str, float], ...], int]]:
    """Reproduce the workflow's stable hidden equation for each eval pack.

    The equation signature is support plus rounded coefficients. Environment
    seeds and initial conditions are deliberately not part of the signature, so
    two packs with the same mathematical reaction are treated as duplicates.
    """
    candidate, ground_truth, pde_numeric = _load_pde_ground_truth_modules()

    templates = ground_truth.load_hidden_gt_templates(
        candidate.DEFAULT_DICTIONARY,
        family=ground_truth_family,
    )
    initial_shape = pde_numeric.InitialConditionShapeConfig.from_mapping(
        {"mode_count_range": list(initial_mode_count_range)}
    )
    initial_amplitude_upper = ground_truth.calibrate_family_initial_amplitude_upper(
        templates,
        min_reaction_terms,
        max_reaction_terms,
        state_abs_limit=state_abs_limit,
    )

    signatures = []
    for descriptor in _eval_pack_descriptors(rows, eval_pack_size):
        environment_seed = descriptor["environment_seed"]
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [environment_seed, pde_numeric.GROUND_TRUTH_STREAM]
            )
        )
        coefficients = {}
        for _ in range(ground_truth.GT_STABILITY_MAX_ATTEMPTS):
            _, coefficients = ground_truth.sample_hidden_reaction(
                rng,
                templates,
                min_reaction_terms,
                max_reaction_terms,
                template_index=descriptor["template_index"],
            )
            if ground_truth.is_stable_reaction_candidate(
                coefficients=coefficients,
                pde_grid_size=257,
                pde_time_steps=2001,
                trajectory_count=eval_pack_size,
                pack_seed=environment_seed,
                initial_amplitude_upper=initial_amplitude_upper,
                initial_condition_shape=initial_shape,
                state_abs_limit=state_abs_limit,
            ):
                break
        signature = tuple(
            (term, float(coefficients[term]))
            for term in candidate.DEFAULT_DICTIONARY
            if term in coefficients
        )
        signatures.append((signature, descriptor["pack_index"]))
    return signatures


def _format_equation_signature(signature: tuple[tuple[str, float], ...]) -> str:
    return " + ".join(f"{coefficient:+.2f}*{term}" for term, coefficient in signature)


def validate_eval_equation_disjointness(
    new_rows: list[dict],
    exclude_eval_paths: list[str],
    eval_pack_size: int,
    ground_truth_family: str,
    min_reaction_terms: int,
    max_reaction_terms: int,
    state_abs_limit: float,
    initial_mode_count_range: tuple[int, int],
) -> None:
    """Fail generation when exact hidden equations repeat internally or historically."""
    signature_args = {
        "eval_pack_size": eval_pack_size,
        "ground_truth_family": ground_truth_family,
        "min_reaction_terms": min_reaction_terms,
        "max_reaction_terms": max_reaction_terms,
        "state_abs_limit": state_abs_limit,
        "initial_mode_count_range": initial_mode_count_range,
    }
    new_records = _hidden_equation_signatures(new_rows, **signature_args)
    new_counts = Counter(signature for signature, _ in new_records)
    internal_duplicates = [signature for signature, count in new_counts.items() if count > 1]
    if internal_duplicates:
        rendered = "; ".join(
            _format_equation_signature(signature) for signature in internal_duplicates[:5]
        )
        raise ValueError(f"New eval set contains duplicate hidden equations: {rendered}")

    excluded_signatures = set()
    excluded_pack_count = 0
    for excluded_path in exclude_eval_paths:
        parquet_path = _resolve_test_parquet(excluded_path)
        old_rows = pd.read_parquet(parquet_path).to_dict("records")
        old_records = _hidden_equation_signatures(old_rows, **signature_args)
        excluded_signatures.update(signature for signature, _ in old_records)
        excluded_pack_count += len(old_records)

    overlaps = sorted(set(new_counts) & excluded_signatures, key=repr)
    if overlaps:
        rendered = "; ".join(
            _format_equation_signature(signature) for signature in overlaps[:5]
        )
        raise ValueError(
            f"New eval set overlaps {len(overlaps)} excluded hidden equations: {rendered}"
        )
    print(
        "Verified hidden-equation disjointness: "
        f"{len(new_records)} new packs are unique and disjoint from "
        f"{excluded_pack_count} excluded packs"
    )


def build_stratified_eval_template_assignments(
    test_size: int,
    eval_pack_size: int | None,
    eval_template_count: int | None,
) -> list[dict] | None:
    """Assign an equal number of fixed eval packs to every template.

    Training rows remain pack-agnostic.  This optional metadata is intended only
    for a fixed scientific benchmark whose row order and pack size are held
    constant across checkpoints.  The metadata is consumed inside the PDE
    environment and is never rendered into the model prompt.
    """
    if eval_pack_size is None and eval_template_count is None:
        return None
    if eval_pack_size is None or eval_template_count is None:
        raise ValueError(
            "eval_pack_size and eval_template_count must be provided together"
        )
    if eval_pack_size <= 0 or eval_template_count <= 0:
        raise ValueError("eval pack size and template count must be positive")
    if test_size % eval_pack_size != 0:
        raise ValueError(
            f"test_size={test_size} must be divisible by eval_pack_size={eval_pack_size}"
        )

    pack_count = test_size // eval_pack_size
    if pack_count % eval_template_count != 0:
        raise ValueError(
            f"eval pack count {pack_count} must be divisible by "
            f"eval_template_count={eval_template_count} for balanced coverage"
        )

    assignments = []
    for pack_index in range(pack_count):
        template_index = pack_index % eval_template_count
        template_instance_index = pack_index // eval_template_count
        assignment = {
            "eval_pack_index": pack_index,
            "ground_truth_template_index": template_index,
            "ground_truth_template_instance": template_instance_index,
        }
        assignments.extend(dict(assignment) for _ in range(eval_pack_size))
    return assignments


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate PDE discovery CoD dataset")
    parser.add_argument("--local_dir", default=DEFAULT_DATA_PATH)
    parser.add_argument("--train_size", type=int, default=50000)
    parser.add_argument("--test_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--eval_pack_size",
        type=int,
        default=None,
        help="Fixed CoD pack size for an optional stratified test split.",
    )
    parser.add_argument(
        "--eval_template_count",
        type=int,
        default=None,
        help=(
            "Number of hidden templates to cover equally in the stratified test "
            "split. Must be used with --eval_pack_size."
        ),
    )
    parser.add_argument(
        "--exclude_eval_path",
        action="append",
        default=[],
        help=(
            "Existing stratified eval directory or test.parquet whose exact hidden "
            "equations must not occur in the new test split. May be repeated."
        ),
    )
    parser.add_argument(
        "--online_eval_dir",
        default=None,
        help=(
            "Optional directory for a smaller, deterministic online-eval subset "
            "selected from the full stratified test split."
        ),
    )
    parser.add_argument(
        "--online_eval_instances_per_template",
        type=int,
        default=None,
        help=(
            "Number of the first equation instances to retain per hidden template "
            "in --online_eval_dir. Must be used with --online_eval_dir."
        ),
    )
    parser.add_argument(
        "--eval_ground_truth_family",
        default="physical_full_eval_hard.json",
        help="Ground-truth family used to reproduce equations for overlap checks.",
    )
    parser.add_argument("--eval_min_reaction_terms", type=int, default=1)
    parser.add_argument("--eval_max_reaction_terms", type=int, default=3)
    parser.add_argument("--eval_state_abs_limit", type=float, default=3.0)
    parser.add_argument(
        "--eval_initial_mode_count_range",
        type=int,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(1, 2),
    )
    parser.add_argument(
        "--test_only",
        action="store_true",
        help="Write only test.parquet, avoiding mixed-schema dataset directories.",
    )
    args = parser.parse_args()

    train_data, test_data = prepare_pde_discovery_data(
        data_path=args.local_dir,
        train_size=args.train_size,
        test_size=args.test_size,
        seed=args.seed,
        eval_pack_size=args.eval_pack_size,
        eval_template_count=args.eval_template_count,
        exclude_eval_paths=args.exclude_eval_path,
        eval_ground_truth_family=args.eval_ground_truth_family,
        eval_min_reaction_terms=args.eval_min_reaction_terms,
        eval_max_reaction_terms=args.eval_max_reaction_terms,
        eval_state_abs_limit=args.eval_state_abs_limit,
        eval_initial_mode_count_range=tuple(args.eval_initial_mode_count_range),
        save_train_split=not args.test_only,
        online_eval_data_path=args.online_eval_dir,
        online_eval_instances_per_template=args.online_eval_instances_per_template,
    )

    print(f"\nTrain rows: {len(train_data)}")
    print(f"Test rows: {len(test_data)}")
    if train_data:
        print(f"Sample: {train_data[0]}")
