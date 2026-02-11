import copy
import os
import subprocess
import sys
import tempfile
from typing import List

import streamlit as st
import yaml
from streamlit.web import cli

from trinity.algorithm import ALGORITHM_TYPE
from trinity.algorithm.advantage_fn import ADVANTAGE_FN
from trinity.algorithm.entropy_loss_fn import ENTROPY_LOSS_FN
from trinity.algorithm.kl_fn import KL_FN
from trinity.algorithm.policy_loss_fn import POLICY_LOSS_FN
from trinity.algorithm.sample_strategy import SAMPLE_STRATEGY
from trinity.common.constants import StorageType
from trinity.manager.config_registry import CONFIG_GENERATORS
from trinity.manager.config_registry.buffer_config_manager import (
    get_train_batch_size,
    parse_priority_fn_args,
)
from trinity.manager.config_registry.trainer_config_manager import use_critic
from trinity.utils.plugin_loader import load_plugins

register_map = {
    "sample_strategy": SAMPLE_STRATEGY,
    "policy_loss_fn": POLICY_LOSS_FN,
    "advantage_fn": ADVANTAGE_FN,
    "kl_loss_fn": KL_FN,
    "kl_penalty_fn": KL_FN,
    "entropy_loss_fn": ENTROPY_LOSS_FN,
}


class ConfigManager:
    def __init__(self):
        if "_init_config_manager" not in st.session_state:
            self.reset_session_state()
            load_plugins()
        self.unfinished_fields = set()
        CONFIG_GENERATORS.set_unfinished_fields(self.unfinished_fields)
        st.set_page_config(page_title="Trinity-RFT Config Generator", page_icon=":robot:")
        st.title("Trinity-RFT Config Generator")
        self.maintain_session_state()
        mode = st.pills(
            "Select Mode",
            options=["Beginner Mode", "Expert Mode"],
            default="Beginner Mode",
            label_visibility="collapsed",
        )
        if mode == "Beginner Mode":
            self.beginner_mode()
        else:
            self.expert_mode()
        if "config_generated" not in st.session_state:
            st.session_state.config_generated = False
        if "is_running" not in st.session_state:
            st.session_state.is_running = False
        self.generate_config()

    @staticmethod
    def run(port: int):
        config_manager_path = os.path.abspath(__file__)
        sys.argv = [
            "streamlit",
            "run",
            config_manager_path,
            "--server.port",
            str(port),
            "--server.fileWatcherType",
            "none",
        ]
        sys.exit(cli.main())

    def reset_session_state(self):
        st.session_state["_init_config_manager"] = True
        for key, value in CONFIG_GENERATORS.default_config.items():
            st.session_state[key] = value

    def maintain_session_state(self):
        st.session_state["_init_config_manager"] = True
        for key in CONFIG_GENERATORS.default_config:
            st.session_state[key] = st.session_state[key]

        def maintain_list_state(prefix, key_list):
            last_idx, del_num = 0, 0
            for idx in range(st.session_state[f"_{prefix}s_num"]):
                if st.session_state.get(f"{prefix}_{idx}_del_flag", False):
                    del_num += 1
                    continue
                for key in key_list:
                    full_key = f"{prefix}_{idx}_{key}"
                    last_full_key = f"{prefix}_{last_idx}_{key}"
                    st.session_state[last_full_key] = st.session_state[full_key]
                last_idx += 1
            st.session_state[f"_{prefix}s_num"] -= del_num

        self.eval_dataset_keys = [
            "name",
            "path",
            "split",
            "subset_name",
            "prompt_key",
            "response_key",
            "temperature",
            "logprobs",
            "n",
        ]
        maintain_list_state("eval_taskset", self.eval_dataset_keys)

        self.inference_model_keys = [
            "model_path",
            "engine_num",
            "tensor_parallel_size",
            "enforce_eager",
            "enable_prefix_caching",
            "enable_chunked_prefill",
            "gpu_memory_utilization",
            "dtype",
            "seed",
            "enable_thinking",
            "enable_history",
            "enable_openai_api",
            "enable_auto_tool_choice",
            "tool_call_parser",
            "reasoning_parser",
        ]
        maintain_list_state("auxiliary_model", self.inference_model_keys)

    def get_configs(self, *config_names: str, columns_spec: List[int] = None):
        CONFIG_GENERATORS.get_configs(*config_names, columns_spec=columns_spec)

    def beginner_mode(self):
        st.subheader("Global Config")
        self.get_configs("project", "exp_name")
        self.get_configs("checkpoint_root_dir")
        self.get_configs("monitor_type", "log_level", "save_interval")

        st.subheader("Model Config")
        self.get_configs("model_path", "max_model_len", columns_spec=[3, 1])

        st.subheader("Algorithm Config")
        self.get_configs("algorithm_type", "repeat_times", "actor_lr", "critic_lr")

        st.subheader("Dataset Config")
        if st.session_state["algorithm_type"] not in ("dpo", "sft"):
            self.get_configs("taskset_path", "explore_batch_size", columns_spec=[3, 1])
        else:
            self.get_configs("experience_buffer_path", "train_batch_size", columns_spec=[3, 1])
        if st.session_state["algorithm_type"] == "dpo":
            self.get_configs("dpo_dataset_kwargs")
        elif st.session_state["algorithm_type"] == "sft":
            self.get_configs("sft_dataset_kwargs")
        else:
            self.get_configs("taskset_args")
        self.get_configs("default_workflow_type", "default_reward_fn_type")
        self.get_configs("total_epochs", "total_steps")

        st.subheader("Resource Config")
        self.get_configs("node_num", "gpu_per_node")
        self.get_configs("engine_num", "tensor_parallel_size")
        self.get_configs("trainer_gpu_num_display", "actor_ulysses_sequence_parallel_size")

        if st.session_state["algorithm_type"] not in ("dpo", "sft"):
            st.subheader("Synchronizer Config")
            st.caption("Synchronization between trainer and explorer.")
            self.get_configs("sync_method", "sync_style", "sync_interval")

    def _expert_model_part(self):
        self.get_configs("project", "exp_name")

        self.get_configs("model_path")
        self.get_configs("critic_model_path")

        self.get_configs("checkpoint_root_dir")

        self.get_configs("monitor_type", "node_num", "gpu_per_node")
        self.get_configs("max_response_tokens", "max_model_len")

    def _expert_buffer_part(self):
        self.get_configs("total_epochs", "total_steps", "explore_batch_size", "train_batch_size")

        self.get_configs(
            "default_workflow_type", "default_eval_workflow_type", "default_reward_fn_type"
        )

        if st.session_state["algorithm_type"] == "dpo":
            with st.expander("DPO Dataset Configs", expanded=True):
                self.get_configs("experience_buffer_path")
                self.get_configs("storage_type")
                self.get_configs("dpo_dataset_kwargs")
        elif st.session_state["algorithm_type"] == "sft":
            with st.expander("SFT Dataset Configs", expanded=True):
                self.get_configs("experience_buffer_path")
                self.get_configs("storage_type")
                self.get_configs("sft_dataset_kwargs")
        else:
            with st.expander("Taskset Configs", expanded=True):
                self.get_configs("taskset_path")
                self.get_configs("taskset_args")

        with st.expander("Eval Tasksets Configs", expanded=True):
            self.get_configs("eval_tasksets")

        if st.session_state["algorithm_type"] not in ("dpo", "sft"):
            with st.expander("Experience Buffer Configs", expanded=True):
                self.get_configs("storage_type")
                self.get_configs("experience_buffer_path")
                self.get_configs("enable_replay_buffer")
                self.get_configs("reuse_cooldown_time", "priority_fn")
                self.get_configs("priority_fn_args")

        # TODO: used for SQL storage
        # self.buffer_advanced_tab = st.expander("Advanced Config")
        # with self.buffer_advanced_tab:
        #     self.get_configs("buffer_max_retry_times", "max_retry_interval")

    def _expert_explorer_part(self):
        self.get_configs("sync_method", "sync_style", "sync_interval", "sync_timeout")

        self.get_configs("runner_per_model", "eval_interval")

        with st.expander("Rollout Model Config", expanded=True):
            self.get_configs("engine_type", "engine_num", "tensor_parallel_size")

            self.get_configs("gpu_memory_utilization", "dtype", "seed")

            self.get_configs("enforce_eager", "enable_prefix_caching", "enable_chunked_prefill")

            self.get_configs("enable_thinking", "enable_history")
            self.get_configs("enable_openai_api", "enable_auto_tool_choice")
            self.get_configs("tool_call_parser", "reasoning_parser")

        with st.expander("Auxiliary Models", expanded=True):
            self.get_configs("auxiliary_models")

    def _expert_trainer_part(self):
        self.get_configs("algorithm_type", "repeat_times", "save_interval")
        self.get_configs("policy_loss_fn", "advantage_fn", "sample_strategy")
        self.get_configs("kl_penalty_fn", "kl_loss_fn", "kl_coef_in_kl_loss_fn")
        self.get_configs("entropy_loss_fn", "entropy_coef_in_entropy_loss_fn")

        with st.expander("Advanced Algorithm Config"):
            algorithm = ALGORITHM_TYPE.get(st.session_state["algorithm_type"])
            default_config = algorithm.default_config()
            config_key_list = []
            for key in default_config.keys():
                value = st.session_state[key]
                if key == "repeat_times":
                    continue
                default_args = register_map[key].get(value).default_args()
                for sub_key in default_args.keys():
                    full_key = sub_key + "_in_" + key
                    if full_key in ("kl_coef_in_kl_loss_fn", "entropy_coef_in_entropy_loss_fn"):
                        continue
                    config_key_list.append(full_key)

            idx = 0
            while idx < len(config_key_list):
                delta = 3 if len(config_key_list) - idx != 4 else 2
                key_list = config_key_list[idx : idx + delta]
                idx += delta
                self.get_configs(*key_list)

        self.get_configs("enable_preview")

        if st.session_state["trainer_type"] == "verl":
            self._expert_verl_trainer_part()

    def _expert_verl_training_part(self):
        st.subheader("RL Training Config")
        self.get_configs("training_args")

        self.get_configs("ppo_epochs", "training_strategy", "resume_mode", "impl_backend")

        self.get_configs("resume_from_path")

        if st.session_state["training_strategy"] == "fsdp":
            self.get_configs("param_offload", "optimizer_offload", "forward_prefetch")
        elif st.session_state["training_strategy"] == "fsdp2":
            self.get_configs("offload_policy", "reshard_after_forward")
        elif st.session_state["training_strategy"] == "megatron":
            with st.expander("Megatron Config"):
                self.get_configs("param_offload", "grad_offload", "optimizer_offload")
                self.get_configs(
                    "tensor_model_parallel_size",
                    "pipeline_model_parallel_size",
                    "virtual_pipeline_model_parallel_size",
                )
                self.get_configs(
                    "expert_model_parallel_size",
                    "expert_tensor_parallel_size",
                    "context_parallel_size",
                )
                self.get_configs(
                    "sequence_parallel",
                    "use_distributed_optimizer",
                    "use_dist_checkpointing",
                    "use_mbridge",
                )
                self.get_configs("dist_checkpointing_path")
                self.get_configs(
                    "recompute_granularity", "recompute_method", "recompute_num_layers"
                )
                self.get_configs("recompute_modules")

        with st.expander("Advanced Config"):
            self.get_configs("critic_warmup")

            self.get_configs("default_hdfs_dir")

            self.get_configs("del_local_ckpt_after_load")

            self.get_configs("max_actor_ckpt_to_keep", "max_critic_ckpt_to_keep")

    def _expert_verl_actor_part(self):
        st.subheader("Actor Model Config")

        self.get_configs("actor_lr", "actor_lr_scheduler_type", "actor_lr_warmup_steps_ratio")

        self.get_configs("actor_grad_clip", "actor_ulysses_sequence_parallel_size")

        self.get_configs(
            "actor_ppo_micro_batch_size_per_gpu",
            "ref_log_prob_micro_batch_size_per_gpu",
            "actor_ppo_max_token_len_per_gpu",
        )

        self.get_configs("actor_entropy_from_logits_with_chunking", "actor_entropy_checkpointing")

        self.get_configs("actor_load_checkpoint")

        self.get_configs("actor_save_checkpoint")

    def _expert_verl_critic_part(self):
        st.subheader("Critic Model Config")
        self.get_configs(
            "critic_ppo_micro_batch_size_per_gpu", "critic_ulysses_sequence_parallel_size"
        )

        self.get_configs("critic_lr", "critic_lr_scheduler_type", "critic_lr_warmup_steps_ratio")

        self.get_configs("critic_grad_clip", "critic_cliprange_value")
        self.get_configs("critic_load_checkpoint", "critic_save_checkpoint")

    def _expert_verl_trainer_part(self):
        name2func = {
            "RL Training Config": self._expert_verl_training_part,
            "Actor and Ref Config": self._expert_verl_actor_part,
        }
        if use_critic():
            name2func["Critic Config"] = self._expert_verl_critic_part

        tabs = st.tabs([name for name in name2func])
        for tab, func in zip(tabs, name2func.values()):
            with tab:
                func()

    def expert_mode(self):
        tab2func = {
            "Model": self._expert_model_part,
            "Buffer": self._expert_buffer_part,
            "Explorer and Synchronizer": self._expert_explorer_part,
            "Trainer": self._expert_trainer_part,
        }
        if st.session_state["mode"] == "train":
            del tab2func["Explorer and Synchronizer"]
        tabs = st.tabs(list(tab2func.keys()))
        for tab, func in zip(tabs, tab2func.values()):
            with tab:
                func()

    def _generate_verl_config(self):
        balance_batch = "balance_batch" in st.session_state["training_args"]
        enable_gradient_checkpointing = (
            "gradient_checkpointing" in st.session_state["training_args"]
        )
        use_remove_padding = "remove_padding" in st.session_state["training_args"]
        use_dynamic_bsz = "dynamic_bsz" in st.session_state["training_args"]
        use_fused_kernels = "use_fused_kernels" in st.session_state["training_args"]

        if st.session_state["training_strategy"] == "fsdp":
            distribution_config = {
                "fsdp_config": {
                    "fsdp_size": -1,
                    "wrap_policy": {"min_num_params": 0},
                    "param_offload": st.session_state["param_offload"],
                    "optimizer_offload": st.session_state["optimizer_offload"],
                    "forward_prefetch": st.session_state["forward_prefetch"],
                }
            }
        elif st.session_state["training_strategy"] == "fsdp2":
            distribution_config = {
                "fsdp_config": {
                    "fsdp_size": -1,
                    "offload_policy": st.session_state["offload_policy"],
                    "reshard_after_forward": st.session_state["reshard_after_forward"],
                }
            }
        elif st.session_state["training_strategy"] == "megatron":
            distribution_config = {
                "megatron": {
                    "param_offload": st.session_state["param_offload"],
                    "grad_offload": st.session_state["grad_offload"],
                    "optimizer_offload": st.session_state["optimizer_offload"],
                    "tensor_model_parallel_size": st.session_state["tensor_model_parallel_size"],
                    "pipeline_model_parallel_size": st.session_state[
                        "pipeline_model_parallel_size"
                    ],
                    "virtual_pipeline_model_parallel_size": st.session_state[
                        "virtual_pipeline_model_parallel_size"
                    ],
                    "expert_model_parallel_size": st.session_state["expert_model_parallel_size"],
                    "expert_tensor_parallel_size": st.session_state["expert_tensor_parallel_size"],
                    "context_parallel_size": st.session_state["context_parallel_size"],
                    "sequence_parallel": st.session_state["sequence_parallel"],
                    "use_distributed_optimizer": st.session_state["use_distributed_optimizer"],
                    "use_dist_checkpointing": st.session_state["use_dist_checkpointing"],
                    "dist_checkpointing_path": st.session_state["dist_checkpointing_path"],
                    "seed": st.session_state["seed"],
                    # TODO: override_ddp_config
                    "override_transformer_config": {
                        "recompute_granularity": st.session_state["recompute_granularity"],
                        "recompute_modules": st.session_state["recompute_modules"],
                        "recompute_method": st.session_state["recompute_method"],
                        "recompute_num_layers": st.session_state["recompute_num_layers"],
                    },
                    "use_mbridge": st.session_state["use_mbridge"],
                }
            }
        else:
            distribution_config = {}

        ppo_max_token_len_per_gpu = (
            st.session_state["repeat_times"] * st.session_state["max_model_len"]
        )

        trainer_config = {
            "actor_rollout_ref": {
                "model": {
                    "external_lib": None,
                    "override_config": {},
                    "enable_gradient_checkpointing": enable_gradient_checkpointing,
                    "use_remove_padding": use_remove_padding,
                    "use_fused_kernels": use_fused_kernels,
                },
                "actor": {
                    "strategy": st.session_state["training_strategy"],
                    "ppo_micro_batch_size_per_gpu": st.session_state[
                        "actor_ppo_micro_batch_size_per_gpu"
                    ],
                    "use_dynamic_bsz": use_dynamic_bsz,
                    "ppo_max_token_len_per_gpu": st.session_state["actor_ppo_max_token_len_per_gpu"]
                    or ppo_max_token_len_per_gpu,
                    "ppo_epochs": st.session_state["ppo_epochs"],
                    "ulysses_sequence_parallel_size": st.session_state[
                        "actor_ulysses_sequence_parallel_size"
                    ],
                    "entropy_from_logits_with_chunking": st.session_state[
                        "actor_entropy_from_logits_with_chunking"
                    ],
                    "entropy_checkpointing": st.session_state["actor_entropy_checkpointing"],
                    "checkpoint": {
                        "load_contents": st.session_state["actor_load_checkpoint"],
                        "save_contents": st.session_state["actor_save_checkpoint"],
                    },
                },
                "ref": {
                    "log_prob_use_dynamic_bsz": use_dynamic_bsz,
                    "log_prob_max_token_len_per_gpu": ppo_max_token_len_per_gpu,
                    "ulysses_sequence_parallel_size": st.session_state[
                        "actor_ulysses_sequence_parallel_size"
                    ],
                    "entropy_from_logits_with_chunking": st.session_state[
                        "actor_entropy_from_logits_with_chunking"
                    ],
                    "entropy_checkpointing": st.session_state["actor_entropy_checkpointing"],
                },
            },
            "critic": {},
            "trainer": {
                "balance_batch": balance_batch,
                "resume_mode": st.session_state["resume_mode"],
                "resume_from_path": st.session_state["resume_from_path"],
                "default_hdfs_dir": st.session_state["default_hdfs_dir"],
                "del_local_ckpt_after_load": st.session_state["del_local_ckpt_after_load"],
                "max_actor_ckpt_to_keep": st.session_state["max_actor_ckpt_to_keep"],
                "max_critic_ckpt_to_keep": st.session_state["max_critic_ckpt_to_keep"],
            },
        }

        trainer_config["actor_rollout_ref"]["actor"].update(copy.deepcopy(distribution_config))
        trainer_config["actor_rollout_ref"]["ref"].update(copy.deepcopy(distribution_config))

        if use_fused_kernels:
            trainer_config["actor_rollout_ref"]["model"]["fused_kernel_options"] = {
                "impl_backend": st.session_state["impl_backend"],
            }

        if use_critic():
            trainer_config["trainer"]["critic_warmup"] = st.session_state["critic_warmup"]
            trainer_config["critic"] = {
                "strategy": st.session_state["training_strategy"],
                "optim": {
                    "lr": st.session_state["critic_lr"],
                    "lr_warmup_steps_ratio": st.session_state["critic_lr_warmup_steps_ratio"],
                    "lr_scheduler_type": st.session_state["critic_lr_scheduler_type"],
                },
                "model": {
                    "override_config": {},
                    "external_lib": None,
                    "enable_gradient_checkpointing": enable_gradient_checkpointing,
                    "use_remove_padding": use_remove_padding,
                },
                "ppo_mini_batch_size": get_train_batch_size(),
                "ppo_micro_batch_size_per_gpu": st.session_state[
                    "critic_ppo_micro_batch_size_per_gpu"
                ],
                "forward_micro_batch_size_per_gpu": st.session_state[
                    "critic_ppo_micro_batch_size_per_gpu"
                ],
                "use_dynamic_bsz": use_dynamic_bsz,
                "ppo_max_token_len_per_gpu": ppo_max_token_len_per_gpu * 2,
                "forward_max_token_len_per_gpu": ppo_max_token_len_per_gpu * 2,
                "ulysses_sequence_parallel_size": st.session_state[
                    "critic_ulysses_sequence_parallel_size"
                ],
                "ppo_epochs": st.session_state["ppo_epochs"],
                "grad_clip": st.session_state["critic_grad_clip"],
                "cliprange_value": st.session_state["critic_cliprange_value"],
                "checkpoint": {
                    "load_contents": st.session_state["critic_load_checkpoint"],
                    "save_contents": st.session_state["critic_save_checkpoint"],
                },
            }
            if st.session_state["training_strategy"] in {"fsdp", "fsdp2"}:
                trainer_config["critic"]["model"].update(copy.deepcopy(distribution_config))
            elif st.session_state["training_strategy"] == "megatron":
                trainer_config["critic"].update(copy.deepcopy(distribution_config))
        else:
            del trainer_config["critic"]
        return trainer_config

    def _gen_algorithm_config(self):
        algorithm_config = {
            "algorithm_type": st.session_state["algorithm_type"],
        }
        algorithm = ALGORITHM_TYPE.get(st.session_state["algorithm_type"])
        default_config = algorithm.default_config()
        current_config = {}
        for key in default_config.keys():
            current_config[key] = value = st.session_state[key]
            if key == "repeat_times":
                continue
            default_args = register_map[key].get(value).default_args()
            args = {}
            for sub_key in default_args.keys():
                full_key = sub_key + "_in_" + key
                args[sub_key] = st.session_state.get(full_key, default_args[sub_key])
            if default_args != args:
                current_config[key + "_args"] = args
        if default_config != current_config:
            algorithm_config.update(current_config)
        optimizer_config = {
            "lr": st.session_state["actor_lr"],
            "lr_warmup_steps_ratio": st.session_state["actor_lr_warmup_steps_ratio"],
            "lr_scheduler_type": st.session_state["actor_lr_scheduler_type"],
        }
        algorithm_config["optimizer"] = optimizer_config
        return algorithm_config

    def _gen_buffer_config(self):
        experience_buffer_path = st.session_state["experience_buffer_path"].strip()
        if st.session_state["algorithm_type"] not in ("dpo", "sft"):
            if (
                not experience_buffer_path
                and st.session_state["storage_type"] == StorageType.SQL.value
            ):
                experience_buffer_path = f"sqlite:///{os.path.join(st.session_state['checkpoint_root_dir'], '.cache', st.session_state['project'], st.session_state['exp_name'])}/data.db"
        else:
            st.session_state["storage_type"] = StorageType.FILE.value

        buffer_config = {
            "batch_size": st.session_state["explore_batch_size"],
            "train_batch_size": st.session_state["train_batch_size"],
            "total_epochs": st.session_state["total_epochs"],
            "total_steps": st.session_state["total_steps"],
            "explorer_input": {},
            "trainer_input": {
                "experience_buffer": {
                    "name": "experience_buffer",
                    "storage_type": st.session_state["storage_type"],
                    "path": experience_buffer_path,
                },
            },
        }
        if not experience_buffer_path:
            del buffer_config["trainer_input"]["experience_buffer"]["path"]
        if st.session_state["train_batch_size"] is None:
            if st.session_state["algorithm_type"] in ("dpo", "sft"):
                buffer_config["train_batch_size"] = (
                    st.session_state["explore_batch_size"] * st.session_state["repeat_times"]
                )
                del buffer_config["batch_size"]
            else:
                del buffer_config["train_batch_size"]
        if st.session_state["algorithm_type"] not in ("dpo", "sft"):
            experience_buffer = buffer_config["trainer_input"]["experience_buffer"]
            experience_buffer["replay_buffer"] = {
                "enable": st.session_state["enable_replay_buffer"],
                "priority_fn": st.session_state["priority_fn"],
                "reuse_cooldown_time": st.session_state["reuse_cooldown_time"],
                "priority_fn_args": parse_priority_fn_args(st.session_state["priority_fn_args"]),
            }

        if st.session_state["mode"] != "train":
            buffer_config["explorer_input"] = {
                "taskset": {
                    "name": "taskset",
                    "storage_type": StorageType.FILE.value,
                    "path": st.session_state["taskset_path"],
                    "split": st.session_state["taskset_split"],
                    "subset_name": st.session_state["taskset_subset_name"],
                    "format": {
                        "prompt_key": st.session_state["taskset_prompt_key"],
                        "response_key": st.session_state["taskset_response_key"],
                    },
                    "rollout_args": {
                        "temperature": st.session_state["temperature"],
                        "logprobs": st.session_state["logprobs"],
                    },
                },
                "eval_tasksets": [],
                "default_workflow_type": st.session_state["default_workflow_type"],
                "default_eval_workflow_type": st.session_state["default_eval_workflow_type"],
                "default_reward_fn_type": st.session_state["default_reward_fn_type"],
            }
            for idx in range(st.session_state["_eval_tasksets_num"]):
                if st.session_state[f"eval_taskset_{idx}_path"].strip():
                    buffer_config["explorer_input"]["eval_tasksets"].append(
                        {
                            "name": st.session_state[f"eval_taskset_{idx}_name"],
                            "path": st.session_state[f"eval_taskset_{idx}_path"],
                            "split": st.session_state[f"eval_taskset_{idx}_split"],
                            "subset_name": st.session_state[f"eval_taskset_{idx}_subset_name"],
                            "format": {
                                "prompt_key": st.session_state[f"eval_taskset_{idx}_prompt_key"],
                                "response_key": st.session_state[
                                    f"eval_taskset_{idx}_response_key"
                                ],
                            },
                            "rollout_args": {
                                "temperature": st.session_state[f"eval_taskset_{idx}_temperature"],
                                "logprobs": st.session_state[f"eval_taskset_{idx}_logprobs"],
                                "n": st.session_state[f"eval_taskset_{idx}_n"],
                            },
                        }
                    )
        else:
            del buffer_config["explorer_input"]

        if st.session_state["algorithm_type"] == "dpo":
            experience_buffer = buffer_config["trainer_input"]["experience_buffer"]
            experience_buffer["split"] = st.session_state["dpo_dataset_train_split"]
            experience_buffer["format"] = {
                "prompt_type": st.session_state["dpo_dataset_prompt_type"],
                "prompt_key": st.session_state["dpo_dataset_prompt_key"],
                "chosen_key": st.session_state["dpo_dataset_chosen_key"],
                "rejected_key": st.session_state["dpo_dataset_rejected_key"],
            }
        elif st.session_state["algorithm_type"] == "sft":
            experience_buffer = buffer_config["trainer_input"]["experience_buffer"]
            experience_buffer["split"] = st.session_state["sft_dataset_train_split"]
            experience_buffer["format"] = {
                "prompt_type": st.session_state["sft_dataset_prompt_type"],
                "prompt_key": st.session_state["sft_dataset_prompt_key"],
                "messages_key": st.session_state["sft_dataset_messages_key"],
            }

        return buffer_config

    def _gen_explorer_config(self):
        explorer_config = {
            "runner_per_model": st.session_state["runner_per_model"],
            "rollout_model": {
                key: st.session_state[key]
                for key in self.inference_model_keys
                if key != "model_path"
                # "chat_template": None,  # TODO: add chat template
            },
            "auxiliary_models": [],
            "eval_interval": st.session_state["eval_interval"],
        }
        for i in range(st.session_state["_auxiliary_models_num"]):
            auxiliary_model_config = {
                key: st.session_state[f"auxiliary_model_{i}_{key}"]
                for key in self.inference_model_keys
            }
            explorer_config["auxiliary_models"].append(auxiliary_model_config)
        return explorer_config

    def generate_config(self):
        if st.session_state["trainer_type"] == "verl":
            trainer_config = self._generate_verl_config()
        else:
            raise ValueError(f"Invalid trainer type: {st.session_state['trainer_type']}")

        if len(self.unfinished_fields) > 0:
            disable_generate = True
            help_messages = (
                f"Please check following fields: `{'`, `'.join(self.unfinished_fields)}`"
            )
        else:
            disable_generate = False
            help_messages = None
        if st.button(
            "Generate Config",
            disabled=disable_generate,
            help=help_messages,
            use_container_width=True,
            icon=":material/create_new_folder:",
        ):
            st.session_state.config_generated = True
            st.session_state.is_running = False
        if st.session_state.config_generated:
            config = {
                "mode": st.session_state["mode"],
                "project": st.session_state["project"],
                "name": st.session_state["exp_name"],
                "checkpoint_root_dir": st.session_state["checkpoint_root_dir"],
                "algorithm": self._gen_algorithm_config(),
                "data_processor": {},  # TODO: Add data processor config
                "model": {
                    "model_path": st.session_state["model_path"],
                    "max_prompt_tokens": st.session_state["max_prompt_tokens"],
                    "min_response_tokens": st.session_state["min_response_tokens"],
                    "max_response_tokens": st.session_state["max_response_tokens"],
                    "max_model_len": st.session_state["max_model_len"],
                },
                "cluster": {
                    "node_num": st.session_state["node_num"],
                    "gpu_per_node": st.session_state["gpu_per_node"],
                },
                "buffer": self._gen_buffer_config(),
                "explorer": self._gen_explorer_config(),
                "trainer": {
                    "trainer_type": st.session_state["trainer_type"],
                    "save_interval": st.session_state["save_interval"],
                    "enable_preview": st.session_state["enable_preview"],
                    "grad_clip": st.session_state["actor_grad_clip"],
                    "trainer_config": trainer_config,
                },
                "monitor": {
                    "monitor_type": st.session_state["monitor_type"],
                },
                "synchronizer": {
                    "sync_method": st.session_state["sync_method"],
                    "sync_style": st.session_state["sync_style"],
                    "sync_interval": st.session_state["sync_interval"],
                    "sync_timeout": st.session_state["sync_timeout"],
                },
                "log": {
                    "level": st.session_state["log_level"],
                },
            }

            if use_critic():
                config["model"]["critic_model_path"] = (
                    st.session_state["critic_model_path"].strip()
                    if st.session_state["critic_model_path"].strip()
                    else st.session_state["model_path"]
                )

            st.session_state.config_generated = True
            st.subheader("Generated Config File")
            # buttons = st.container()
            # save_btn, run_btn = buttons.columns(2, vertical_alignment="bottom")
            yaml_config = yaml.dump(config, allow_unicode=True, sort_keys=False)
            # save_btn.download_button(
            #     "Save",
            #     data=yaml_config,
            #     file_name=f"{config['project']}-{config['name']}.yaml",
            #     mime="text/plain",
            #     icon=":material/download:",
            #     use_container_width=True,
            # )
            # run_btn.button(
            #     "Run",
            #     on_click=self.run_config,
            #     args=(
            #         buttons,
            #         yaml_config,
            #     ),
            #     icon=":material/terminal:",
            #     use_container_width=True,
            #     disabled=st.session_state.is_running,
            # )
            st.download_button(
                "Save",
                data=yaml_config,
                file_name=f"{config['project']}-{config['name']}.yaml",
                mime="text/plain",
                icon=":material/download:",
                use_container_width=True,
            )
            st.code(yaml_config, language="yaml")

    def run_config(self, parent, yaml_config: str) -> None:
        st.session_state.is_running = True

        import ray

        # first check if ray is running
        ray_status = subprocess.run(
            ["ray", "status"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if ray_status.returncode != 0:
            parent.warning(
                "Ray cluster is not running. Please start Ray first using `ray start --head`."
            )
            return
        context = ray.init(ignore_reinit_error=True)
        dashboard_url = context.dashboard_url
        # save config to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmpfile:
            tmpfile.write(yaml_config)
            tmpfile_path = tmpfile.name

        # submit ray job
        try:
            subprocess.run(
                [
                    "ray",
                    "job",
                    "submit",
                    "--no-wait",
                    "--",
                    "python",
                    "-m",
                    "trinity.cli.launcher",
                    "run",
                    "--config",
                    tmpfile_path,
                ],
                text=True,
                capture_output=True,
                check=True,
            )
            parent.success(
                f"Job submitted successfully!\n\n"
                f"View progress in the Ray Dashboard: http://{dashboard_url}",
                icon="✅",
            )
        except subprocess.CalledProcessError as e:
            parent.error(f"Failed to submit job:\n\n{e.stderr}", icon="❌")
            st.session_state.is_running = False


if __name__ == "__main__":
    config_manager = ConfigManager()
