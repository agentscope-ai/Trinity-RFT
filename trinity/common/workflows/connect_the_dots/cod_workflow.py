"""Connect-the-Dots (CoD) workflow for cross-task learning and generalization."""

import asyncio
import hashlib
import json
import os
import random
from copy import deepcopy
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple, Union

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows.connect_the_dots.cod_utils import CoDPrompts, compute_part_summary
from trinity.common.workflows.workflow import Task, Workflow


def compute_task_hash(task_desc: str) -> str:
    """Compute hash of task description."""
    return hashlib.md5(task_desc.encode()).hexdigest()[:16]


def get_task_source(task: Task) -> str:
    if task.raw_task is None:
        return ""
    return task.raw_task.get("metadata", {}).get("source", "")

def get_taskset_id(task: Task) -> int:
    """Get taskset_id of a task.
    
    Ref: class TasksetScheduler (renamed to DatasetScheduler later), 
        task.index["taskset_id"] = taskset_id (type: int)
        This is applicable only when there are > 1 tasksets.
    """
    # return task.index["taskset_id"]
    return task.index.get("taskset_id", 0)


def pack_tasks(
    tasks: List[Task],
    pack_size: int,
    cod_workflow_args: dict,
) -> List[Task]:
    """Group tasks by taskset_id, pack for each group, and merge outputs.

    Drop a group if it has fewer than pack_size tasks.
    
    If cod_workflow_args.get("pad_tasks_to_full_pack", False) is True:
    for each group, pad tasks to multiple of pack_size (in a cyclic manner).
    """
    returned_tasks: List[Task] = []
    grouped_tasks = dict()  # taskset_id -> list of original tasks

    # Group tasks by taskset_id
    for task in tasks:
        taskset_id = get_taskset_id(task)
        if taskset_id not in grouped_tasks:
            grouped_tasks[taskset_id] = []
        grouped_tasks[taskset_id].append(task)

    # Logs for debug
    log_count_original = dict()
    log_count_final = dict()
    log_count_pack = dict()

    # Pack tasks for each group, and merge outputs
    for id, task_lst in grouped_tasks.items():
        log_count_original[id] = len(task_lst)

        if len(task_lst) < pack_size:
            log_count_final[id] = 0
            log_count_pack[id] = 0
            continue  # too few tasks, drop this group

        if cod_workflow_args.get("pad_tasks_to_full_pack", False):
            remain_count = len(task_lst) % pack_size
            if remain_count > 0:
                pad_count = pack_size - remain_count
                task_lst = task_lst + task_lst[:pad_count]  # pad in a cyclic manner
        log_count_final[id] = len(task_lst)

        packed_tasks = pack_tasks_from_same_taskset(
            tasks=task_lst,
            pack_size=pack_size,
            cod_workflow_args=cod_workflow_args,
        )
        returned_tasks.extend(packed_tasks)
        log_count_pack[id] = len(packed_tasks)

    for id in log_count_original.keys():
        print(
            f"original task count {log_count_original[id]},",
            f"final task count {log_count_final[id]},",
            f"final pack count {log_count_pack[id]}.",
        )
    
    return returned_tasks


def pack_tasks_from_same_taskset(
    tasks: List[Task],
    pack_size: int,
    cod_workflow_args: dict,
) -> List[Task]:
    packed_tasks = []

    ensure_same_source = cod_workflow_args.get("ensure_same_source", False)
    if ensure_same_source:
        # Certain CoD method (e.g., rollout tree) requires that all tasks in one pack
        # should come from the same source/scenario/domain. To ensure this:
        #   1. Sort tasks by source before slicing into sub-lists; then after slicing,
        #   2.1. Find out the dominating source in sublst, and remove tasks from all other sources;
        #   2.2. Duplicate tasks from this source if necessary, to match the length of original sublst.

        # Step 1 is done here, and Step 2.1 - 2.2 is done in the for-loop below.
        tasks = sorted(tasks, key=lambda task: get_task_source(task))

    for i in range(len(tasks) // pack_size):
        start, end = i * pack_size, (i + 1) * pack_size
        sublst = tasks[start:end]

        if ensure_same_source:
            source_counts = {}
            for task in sublst:
                task_source = get_task_source(task)
                source_counts[task_source] = source_counts.get(task_source, 0) + 1
            dominating_source = max(source_counts, key=source_counts.get)
            filtered_tasks = [task for task in sublst if get_task_source(task) == dominating_source]

            new_sublst = []
            target_len = len(sublst)
            while len(new_sublst) < target_len:
                new_sublst.extend(filtered_tasks)
            sublst = new_sublst[:target_len]

        new_task = deepcopy(sublst[0])
        new_task.workflow = CoDWorkflow
        new_task.workflow_args = deepcopy(cod_workflow_args)

        if new_task.raw_task is None:
            new_task.raw_task = dict()
        new_task.raw_task["aux_tasks"] = sublst[1:]
        new_task.raw_task["all_tasks"] = sublst

        packed_tasks.append(new_task)

    if ensure_same_source:
        random.shuffle(packed_tasks)

    return packed_tasks


class CoDWorkflow(Workflow):
    """A CoD "meta workflow" that can call an existing workflow's run method.

    Assumptions for current version:
        - input task is a pack of multiple tasks in task.raw_task["all_tasks"]
        - task-solving workflow is single-turn
        - task-solving workflow can_reset = True, is_async = True
    """

    can_reset: bool = True
    can_repeat: bool = True
    is_async: bool = True

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
        self.all_tasks: List[Task] = []
        self.all_workflow_instances: List[
            Union[Workflow, None]
        ] = []  # one workflow instance for each task

        self.reset(task)

    def reset(self, cod_task: Task):
        """Reset current CoD workflow, as well as all task-solving workflow instances.

        (Doesn't matter much if we always use max_repeat_times_per_runner = 1)
        """
        self.task = cod_task

        self.all_tasks = []
        if isinstance(cod_task.raw_task, dict) and "all_tasks" in cod_task.raw_task.keys():
            self.all_tasks = cod_task.raw_task["all_tasks"]  # List[Task]
        lst_taskset_id = [str(get_taskset_id(task)) for task in self.all_tasks]

        # Optionally inject pack-level seed into sub-tasks (for per_pack mapping mode in FrozenLakeObscure, etc.)
        # Enable via workflow_args: inject_pack_seed: true
        if cod_task.workflow_args.get("inject_pack_seed", False):
            pack_seed = hash((cod_task.batch_id, cod_task.task_id)) % (2**32)
            for i, task in enumerate(self.all_tasks):
                if task.raw_task is None:
                    task.raw_task = {}
                task.raw_task["pack_seed"] = pack_seed
                task.raw_task["task_idx"] = i
                task.raw_task["pack_size"] = len(self.all_tasks)

        reuse_workflow_instance = cod_task.workflow_args["reuse_workflow_instance"]
        if reuse_workflow_instance is False:
            self.all_workflow_instances = [None for _ in range(len(self.all_tasks))]
            return

        if len(self.all_workflow_instances) == 0:
            for task in self.all_tasks:
                workflow_instance = task.to_workflow(
                    self.model,
                    self.auxiliary_model_wrappers,
                )
                self.all_workflow_instances.append(workflow_instance)
        else:
            for i, task in enumerate(self.all_tasks):
                workflow_instance = self.all_workflow_instances[i]
                # 类型一致且可以 reset 才复用
                if workflow_instance.__class__ == task.workflow and workflow_instance.can_reset:
                    workflow_instance.reset(task)
                else:
                    self.all_workflow_instances[i] = task.to_workflow(
                        self.model,
                        self.auxiliary_model_wrappers,
                    )

    def set_repeat_times(self, repeat_times, run_id_base):
        # TODO: decide whether repeat_times should be always 1
        self.repeat_times = repeat_times
        self.task.rollout_args.n = repeat_times
        self.run_id_base = run_id_base
        for task in self.all_tasks:
            task.rollout_args.n = repeat_times
        for workflow_instance in self.all_workflow_instances:
            if workflow_instance is not None and workflow_instance.can_repeat:
                workflow_instance.set_repeat_times(repeat_times, run_id_base)

    # --- CoD utils ---

    @staticmethod
    def _get_trajectory(task_exps: List[Experience]) -> str:
        """Get trajectory string from experiences.

        Requires workflow to provide trajectory in exp.info["trajectory"].
        Raises KeyError if trajectory is not provided.
        """
        if not task_exps:
            raise ValueError("task_exps is empty, cannot get trajectory")

        last_exp = task_exps[-1]
        if "trajectory" not in last_exp.info:
            raise KeyError(
                "Workflow must provide trajectory in exp.info['trajectory']. "
                "For multi-step workflows, inherit from AsyncCoDMultiStepWorkflow. "
                "For single-turn workflows, set trajectory in run_async() with user + assistant format."
            )
        return last_exp.info["trajectory"]
    
    @staticmethod
    def _get_feedback(task_exps: List[Experience], task_reward: Optional[float]) -> str:
        """Get feedback string from experiences."""
        last_exp = task_exps[-1] if task_exps else None
        if last_exp and "feedback" in last_exp.info:
            feedback = last_exp.info["feedback"]
        else:
            # Fallback for workflows without feedback
            assert task_reward is not None
            feedback = f"Reward: {task_reward} (max: 1.0)"
        return feedback

    def _post_process_hint_generation_exp(
        self,
        exp: Experience,
        subtid: int,
        part_id: int,
        part_name: str,
        repeat_id: int,
        step_id: int,
    ) -> None:
        if exp.metrics is None:
            exp.metrics = {}
        hint = exp.info["hint"]
        key_hint_len_with_suffix = f"hint_len_in_char_{part_name}"
        exp.metrics.update(
            {
                "reward_gen_hint": exp.reward,
                key_hint_len_with_suffix: len(hint),
            }
        )

        exp.eid.task = "_".join(
            [
                "batch",
                str(self.task.batch_id),
                "main",
                str(self.task.task_id),
                "sub",
                str(subtid),
            ]
        )
        exp.eid.run = (
            self.run_id_base * 100 + self.repeat_times * part_id + repeat_id
        )  # TODO: eliminate magic number 100
        exp.eid.step = step_id

    def _post_process_task_solving_exp(
        self,
        exp: Experience,
        subtid: int,
        part_id: int,
        part_name: str,
        repeat_id: int,
        append_part_name_to_task_id: bool = False,
    ) -> None:
        # Only update metrics if exp.metrics is non-empty (i.e., the last exp in a trajectory)
        # This avoids metric explosion (mean@101, mean@102, etc.) when using multi-step workflows
        # !!! TODO: this prevents setting metrics in workflow, better change this if-condition to something more robust
        # !!! TODO: another issue here is that response_len_in_char is only record for last exp of an episode.
        if exp.metrics:
            key_reward_with_suffix = f"reward_{part_name}"
            key_response_len_with_suffix = f"response_len_in_char_{part_name}"
            exp.metrics.update(
                {
                    "reward": exp.reward,
                    key_reward_with_suffix: exp.reward,
                    key_response_len_with_suffix: len(exp.response_text),
                }
            )

        # task / run / step id
        task_id_pieces = [
            "batch",
            str(self.task.batch_id),
            "main",
            str(self.task.task_id),
            "sub",
            str(subtid),
        ]
        if append_part_name_to_task_id:
            # useful for contolling which exps to group in grpo-like advantage calculation
            task_id_pieces.extend(["part", part_name])
        exp.eid.task = "_".join(task_id_pieces)
        exp.eid.run = (
            self.run_id_base * 100 + self.repeat_times * part_id + repeat_id
        )  # TODO: eliminate magic number 100
        # step_id is set by workflow internally (e.g., AsyncCoDMultiStepWorkflow.run_async)

    def _create_sublogs(
        self,
        task: Task,
        exps: List[Experience],
    ) -> Dict[str, Union[str, None, dict, list]]:
        sublogs: Dict[str, Union[str, None, dict, list]] = {}
        # Only include task_desc / truth when they are not None
        if task.task_desc is not None:
            sublogs["task_desc"] = task.task_desc
        if task.truth is not None:
            sublogs["truth"] = task.truth

        # Deduplicate sys_prompts and hints
        sys_prompt_list: List[str] = []
        sys_prompt_map: Dict[str, int] = {}  # content -> index
        hint_list: List[str] = []
        hint_map: Dict[str, int] = {}  # content -> index

        steps: List[dict] = []
        for exp in exps:
            taskid = exp.eid.task
            runid = exp.eid.run
            stepid = exp.eid.step
            sys_prompt = exp.info.get("sys_prompt", "(not logged)")
            user_prompt = exp.info.get("user_prompt", "(not logged)")
            hint = exp.info.get("hint", "(not available)")
            response_text = exp.response_text
            reward = exp.reward

            # Assign deduplicated index for sys_prompt
            if sys_prompt not in sys_prompt_map:
                sys_prompt_map[sys_prompt] = len(sys_prompt_list)
                sys_prompt_list.append(sys_prompt)
            # Assign deduplicated index for hint
            if hint not in hint_map:
                hint_map[hint] = len(hint_list)
                hint_list.append(hint)

            # Simplified step key: strip batch/main/part info (already in dir/file name)
            # Original taskid format: batch_X_main_Y_sub_Z[_part_NAME]
            # Extract sub_Z portion
            sub_part = ""
            parts = taskid.split("_")
            try:
                sub_idx = parts.index("sub")
                # Take sub and its value, stop before "part" if present
                if "part" in parts:
                    part_idx = parts.index("part")
                    sub_part = "_".join(parts[sub_idx:part_idx])
                else:
                    sub_part = "_".join(parts[sub_idx:])
            except ValueError:
                sub_part = taskid  # fallback
            step_key = f"{sub_part}_run_{runid}_step_{stepid}"

            step_entry = {
                "step_key": step_key,
                "sys_prompt_idx": sys_prompt_map[sys_prompt],
                "hint_idx": hint_map[hint],
                "user_prompt": user_prompt,
                "response_text": response_text,
                "reward": reward,
            }
            # Additional fields
            if "exp_type" in exp.info:
                step_entry["exp_type"] = exp.info["exp_type"]
            if "incurred_mean_reward" in exp.info:
                step_entry["incurred_mean_reward"] = exp.info["incurred_mean_reward"]
            if "total_reward" in exp.info:
                step_entry["total_reward"] = exp.info["total_reward"]

            steps.append(step_entry)

        sublogs["sys_prompts"] = sys_prompt_list
        sublogs["hints"] = hint_list
        sublogs["steps"] = steps

        # Group steps into episodes by sys_prompt_idx for easier reading
        episodes: Dict[int, list] = {}
        for step_entry in steps:
            ep_id = step_entry["sys_prompt_idx"]
            if ep_id not in episodes:
                episodes[ep_id] = []
            episodes[ep_id].append(step_entry)
        sublogs["episodes"] = episodes

        return sublogs

    def _append_token_limit_to_prompt(self, prompt: str) -> str:
        """Append token limit instruction to prompt if configured."""
        max_tokens = self.task.workflow_args.get("max_response_tokens_restraint")
        if max_tokens:
            prompt += f"\n\nPlease limit your response to {max_tokens} tokens."
        return prompt

    # --- CoD methods ---

    async def gen_hint_iteratively(
        self,
        prev_hint: str,
        trajectory: str,
        reward: float,
        feedback: str,
    ) -> Tuple[Experience, str]:
        """Generate updated hint based on previous hint and a solved task's trajectory.

        This is used in iterative hint workflows where hints evolve based on
        experience from solving sequential tasks.

        Args:
            prev_hint: Previous hint/instructions (empty string if first iteration)
            trajectory: The solution trajectory (includes task description and model responses)
            reward: The reward received
            feedback: Environment feedback (e.g., final state)

        Returns:
            Tuple of (experience, new_hint)
        """
        hint_example = self.task.workflow_args.get("hint_example", False)
        cod_system_prompt = CoDPrompts.sys_prompt_gen_hint_iteratively(hint_example)
        cod_system_prompt = self._append_token_limit_to_prompt(cod_system_prompt)
        cod_user_prompt = CoDPrompts.user_prompt_gen_hint_iteratively(
            prev_hint=prev_hint,
            trajectory=trajectory,
            reward=reward,
            feedback=feedback,
        )
        messages_gen_hint = [
            {"role": "system", "content": cod_system_prompt},
            {"role": "user", "content": cod_user_prompt},
        ]
        rollout_args = deepcopy(asdict(self.task.rollout_args))
        rollout_args["n"] = 1
        exps = await self.model.chat_async(messages_gen_hint, **rollout_args)
        exp = exps[0]

        # reward penalty for gen_hint sample by response length:
        #   * too short: -hint_penalty_coef if resp_len < len_min_penalty
        #   * too long:  -hint_penalty_coef * linear ramp from len_zero_penalty to len_max_penalty
        hint_penalty_coef = self.task.workflow_args.get("hint_penalty_coef", 0.0)
        len_zero_penalty = self.task.workflow_args.get("len_zero_penalty", None)
        len_max_penalty = self.task.workflow_args.get("len_max_penalty", None)
        len_min_penalty = self.task.workflow_args.get("len_min_penalty", None)
        if (len_zero_penalty is None) or (len_max_penalty is None):
            exp.reward = 0.0
        else:
            assert len_zero_penalty < len_max_penalty, "len_zero_penalty must be smaller than len_max_penalty."
            resp_len = len(exp.tokens) - exp.prompt_length
            if len_min_penalty is not None and resp_len < len_min_penalty:
                exp.reward = -1.0 * hint_penalty_coef
            else:
                exp.reward = -1.0 * hint_penalty_coef * min(1.0, max(0.0, (resp_len - len_zero_penalty) / (len_max_penalty - len_zero_penalty)))

        # parse response and extract hint; overwrite exp.reward if parsing fails
        hint, parse_success = CoDPrompts.extract_hint(exp.response_text)
        if parse_success is False:
            exp.reward = -1.0 * hint_penalty_coef

        if exp.metrics is None:
            exp.metrics = {}
        exp.metrics["hint_parse_success"] = 1.0 * parse_success

        exp.info["sys_prompt"] = cod_system_prompt
        exp.info["user_prompt"] = cod_user_prompt
        exp.info["hint"] = hint
        exp.info["prev_hint"] = prev_hint

        return exp, hint

    def _get_or_create_workflow(self, task: Task, idx: int) -> Workflow:
        """Get or create workflow instance for a task, updating self.all_workflow_instances.

        Args:
            task: The task to create workflow for
            idx: Index in self.all_workflow_instances

        Returns:
            The workflow instance (either reused or newly created)
        """
        workflow_instance = self.all_workflow_instances[idx]
        if workflow_instance is None:
            workflow_instance = task.to_workflow(self.model, self.auxiliary_model_wrappers)
            self.all_workflow_instances[idx] = workflow_instance
        elif workflow_instance.can_reset:
            workflow_instance.reset(task)
        else:
            workflow_instance = task.to_workflow(self.model, self.auxiliary_model_wrappers)
            self.all_workflow_instances[idx] = workflow_instance
        return workflow_instance

    async def solve_task(
        self,
        workflow_instance: Workflow,
        hint: Optional[str] = None,
        icl_examples: Optional[str] = None,
    ) -> List[Experience]:
        """Solve a task and return all exps (supports multi-step workflows).

        Args:
            workflow_instance: Workflow instance to use (must be provided, should be already reset)
            hint: Optional hint to guide the model
            icl_examples: Optional ICL examples to prepend

        Returns:
            List of experiences from solving the task
        """
        if hint is not None:
            workflow_instance.set_hint(hint=hint)
        if icl_examples is not None:
            workflow_instance.set_icl_examples(icl_examples)
        max_tokens = self.task.workflow_args.get("max_response_tokens_restraint")
        if max_tokens:
            workflow_instance.set_max_response_tokens_restraint(max_tokens)
        exps = await workflow_instance.run_async()
        for exp in exps:
            exp.info["hint"] = hint if hint else ""
        return exps

    async def _run_one_iterative_hint_trajectory_e2e(
        self,
        part_id: int,
        part_name: str,
        repeat_id: int,
        trajectory_id: str,
    ) -> Tuple[List[Experience], float]:
        """Run one iterative hint trajectory across all tasks in the pack, for end-to-end meta-training.

        Process: 
            z0 = "Hints: null"  # we might support customizing initial hint later
            -> traj T1 for task x1 
            -> update z1 
            -> traj T2 for task x2 
            -> update z2
            -> traj T3 for task x3

            All samples (including solve_task and gen_hint) will share the same advantage value.

        Assumption:
            exp.info should contain fields "trajectory" and "feedback", as input for hint generation.

        Returns:
            Tuple of (all_exps, total_reward)
            - all_exps: List of all experiences (solve_task exps + gen_hint exps)
            - total_reward: Sum of all task rewards (r1 + r2 + r3 + r4)
        """
        all_exps: List[Experience] = []
        # gen_hint_exps: List[Experience] = []
        task_rewards: List[float] = []
        current_hint = "Hints: null"

        num_tasks = len(self.all_tasks)

        for task_idx in range(num_tasks):
            task = self.all_tasks[task_idx]

            workflow_instance = self._get_or_create_workflow(task, task_idx)

            # (1) Solve task
            task_exps = await self.solve_task(
                workflow_instance=workflow_instance,
                hint=current_hint,
            )

            # Post-process solve_task exps
            for exp in task_exps:
                self._post_process_task_solving_exp(
                    exp=exp,
                    subtid=0,  # !!! ensure all exps in the trajectory have the same eid.task and eid.run
                    part_id=part_id,
                    part_name=part_name,
                    repeat_id=repeat_id,
                )
                exp.info["task_desc"] = task.task_desc
                exp.info["trajectory_id"] = trajectory_id
                exp.info["task_idx"] = task_idx
                exp.info["exp_type"] = "solve_task"

            # Add position-specific reward and format error metrics (on last exp only)
            if task_exps and task_exps[-1].metrics:
                taskset_id = get_taskset_id(task)
                task_exps[-1].metrics[f"reward_{part_name}_taskset_{taskset_id}_pos_{task_idx}"] = task_exps[-1].reward
                if "format_error_termination" in task_exps[-1].metrics:
                    task_exps[-1].metrics[f"format_error_{part_name}_taskset_{taskset_id}_pos_{task_idx}"] = task_exps[-1].metrics["format_error_termination"]

            all_exps.extend(task_exps)

            # Get reward and feedback from last exp
            task_reward = task_exps[-1].reward if task_exps else 0.0
            task_rewards.append(task_reward)

            # Build trajectory string from all responses
            trajectory = self._get_trajectory(task_exps)

            # Get structured feedback from exp.info (set by workflow)
            feedback = self._get_feedback(task_exps, task_reward)

            # (2) Generate updated hint (for all tasks except the last one)
            if task_idx == num_tasks - 1:
                continue

            exp_gen_hint, new_hint = await self.gen_hint_iteratively(
                prev_hint=current_hint,
                trajectory=trajectory,
                reward=task_reward,
                feedback=feedback,
            )

            # Post-process gen_hint exp
            self._post_process_hint_generation_exp(
                exp=exp_gen_hint,
                subtid=0,  # !!! ensure all exps in the trajectory have the same eid.task and eid.run
                part_id=part_id,
                part_name=part_name,
                repeat_id=repeat_id,
                step_id=task_idx,  # use task_idx as step_id for hint. TODO: confirm that step_id make no real effect
            )
            exp_gen_hint.info["trajectory_id"] = trajectory_id
            exp_gen_hint.info["hint_idx"] = task_idx
            exp_gen_hint.info["exp_type"] = "gen_hint"
            # gen_hint_exps.append(exp_gen_hint)
            all_exps.append(exp_gen_hint)

            current_hint = new_hint

        # Calculate total reward (optionally weighted by hint_reward_decay)
        # weight_i = decay^(n-1-i), later tasks weighted more
        # e.g., with decay=0.8 and 4 tasks: [0.8^3, 0.8^2, 0.8^1, 0.8^0]
        # default decay=1.0 means equal weights (equivalent to plain sum)
        task_reward_decay = self.task.workflow_args.get("task_reward_decay", 1.0)
        n = len(task_rewards)
        total_reward = sum(
            r * (task_reward_decay ** (n - 1 - i))
            for i, r in enumerate(task_rewards)
        )

        # Set total reward in exp.info for all exps (for advantage calculation)
        for exp in all_exps:
            exp.info["total_reward"] = total_reward
            exp.info["task_rewards"] = task_rewards

        # Any flagged exp contaminates the shared task_rewards → mask the whole trajectory
        # and strip reward-shaped metrics so wandb doesn't log judge-breakage as "wrong answer".
        if any(e.info.get("judge_format_error", False) for e in all_exps):
            for exp in all_exps:
                exp.info["judge_format_error_in_trajectory"] = True
                if exp.metrics:
                    for k in list(exp.metrics.keys()):
                        if k.startswith("reward") or k.startswith("format_error"):
                            exp.metrics.pop(k, None)

        return all_exps, total_reward

    async def run_iterative_hint_e2e(
        self,
        part_id: int,
        part_name: str,
        subtid: int,
    ) -> List[Experience]:
        """Run iterative hint for end-to-end meta-training.

        Only runs when subtid == 0 (handles all tasks in one call).
        """
        if subtid != 0:
            return []

        all_exps: List[Experience] = []

        for i in range(self.repeat_times):
            trajectory_id = f"batch_{self.task.batch_id}_task_{self.task.task_id}_runidbase_{self.run_id_base}_repeat_{i}"

            exps, total_reward = await self._run_one_iterative_hint_trajectory_e2e(
                part_id=part_id,
                part_name=part_name,
                repeat_id=i,
                trajectory_id=trajectory_id,
            )

            all_exps.extend(exps)

        return all_exps


    def _get_cod_method_mapping(self):
        mapping = {
            "iterative_hint_e2e": self.run_iterative_hint_e2e,
        }
        return mapping

    # --- main cod workflow run method ---

    async def run_parts_sequential(self) -> List[Experience]:
        """Run diverse CoD methods, record metrics, save logs to json files, return experiences.

        For example,
            Part 0: solve directly for each task (baseline method)
            Part 1: for each task, first generate hint with aux tasks, then generate solution with hint
            Part 2: for each task, first generate hint with aux items, then generate solution with hint

        Things to keep in mind:
            + Set task id and step id for each exp carefully, used later by multi-step grpo advantage
            + Separate reward metric names that specify rollout methods
            + One workflow run -> save logs to one folder, containing one or multiple json files
        """

        cod_method_keys = self.task.workflow_args["activated_cod_methods"]
        cod_method_mapping = self._get_cod_method_mapping()

        assert len(cod_method_keys) >= 1

        returned_exps: List[Experience] = []

        # --- prepare for logging ---
        cod_log_interval = self.task.workflow_args.get("cod_log_interval", 20)
        if self.task.is_eval:
            explore_step_num = (
                int(str(self.task.batch_id).split("/")[0]) + 1
            )  # (eval) step num starts from 0
            log_prefix = "eval_" + str(self.task.batch_id).split("/")[1]
        else:
            explore_step_num = int(self.task.batch_id)  # (explore) step num starts from 1
            log_prefix = "rollout"
        self.save_cod_logs: bool = (explore_step_num - 1) % cod_log_interval == 0

        save_path = None
        if self.save_cod_logs:
            log_dir = self.task.workflow_args["log_dir"]
            exp_name = self.task.workflow_args["exp_name"]
            # batch_id = str(self.task.batch_id)
            task_id = str(self.task.task_id)
            run_id_base = str(self.run_id_base)
            subfolder = f"{log_prefix}_step_{explore_step_num}_task_{task_id}_runidbase_{run_id_base}"
            save_path = os.path.join(log_dir, exp_name, subfolder)
            # For eval, skip saving if subfolder already exists
            if self.task.is_eval and os.path.exists(save_path):
                self.save_cod_logs = False
            else:
                os.makedirs(save_path, exist_ok=True)

        # to be used in later parts that require auxiliary trajectories; aligned with self.all_tasks
        self.all_aux_exps: List[Experience] = []

        # Accumulate per-part summary stats for summary.json
        summary_parts: Dict[str, dict] = {}

        # --- one part per activated cod method ---
        for part_id, part_name in enumerate(cod_method_keys):
            cod_fn = cod_method_mapping[part_name]
            exps_part = []
            logs_part = dict()

            async def helper_run_one_task_part(subtid: int):
                task = self.all_tasks[subtid]
                exps = await cod_fn(
                    part_id=part_id,
                    part_name=part_name,
                    subtid=subtid,
                )
                sublogs = self._create_sublogs(task=task, exps=exps)
                return exps, sublogs

            # solve multiple tasks in parallel
            results_part = await asyncio.gather(
                *[helper_run_one_task_part(subtid) for subtid in range(len(self.all_tasks))]
            )
            for subtid, rst in enumerate(results_part):
                exps, sublogs = rst[0], rst[1]
                exps_part.extend(exps)
                logs_part[f"subtid_{subtid}"] = sublogs

            returned_exps.extend(exps_part)
            if self.save_cod_logs and save_path:
                file_name = f"{part_id}_{part_name}.json"
                file_path = os.path.join(save_path, file_name)
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(logs_part, f, indent=4, ensure_ascii=False)
                summary_parts[part_name] = compute_part_summary(exps_part)

        # --- Write summary.json ---
        if self.save_cod_logs and save_path and summary_parts:
            summary = {
                "rollout_step": explore_step_num,
                "parts": summary_parts,
            }
            summary_path = os.path.join(save_path, "summary.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

        return returned_exps

    async def run_parts_parallel(self) -> List[Experience]:
        raise NotImplementedError

    async def run_async(self) -> List[Experience]:
        """TODO: implement run_parts_parallel if it can bring sufficient gains in efficiency.

        reuse_workflow_instance = self.task.workflow_args["reuse_workflow_instance"]

        if reuse_workflow_instance:
            # workflow instance is stateful and resettable, hence need to execute workflow runs sequentially
            exps = await self.run_parts_sequential()
        else:
            # a new workflow instance is created for each workflow run, hence multiple runs can be executed in parallel
            exps = await self.run_parts_parallel()
        """
        exps = await self.run_parts_sequential()
        return exps
