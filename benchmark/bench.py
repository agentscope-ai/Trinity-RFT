import argparse
import importlib
import os
import subprocess
import sys
import time

import torch
import torch.distributed as dist
import yaml

from trinity.algorithm import ALGORITHM_TYPE
from trinity.common.constants import MODEL_PATH_ENV_VAR, SyncStyle
from trinity.utils.dlc_utils import get_dlc_env_vars


def set_engine_num(config, args):
    config["cluster"]["node_num"] = args.node_num
    config["cluster"]["gpu_per_node"] = args.gpu_per_node
    batch_size = config["buffer"]["batch_size"] * config["algorithm"]["repeat_times"]
    if config["mode"] == "train":
        return

    if args.vllm_tp_size is not None:
        config["explorer"]["rollout_model"]["tensor_parallel_size"] = args.vllm_tp_size
    tensor_parallel_size = config["explorer"]["rollout_model"]["tensor_parallel_size"]

    if args.vllm_engine_num is not None:
        config["explorer"]["rollout_model"]["engine_num"] = args.vllm_engine_num
    else:  # auto set engine_num
        opt_explorer_num, opt_ratio_diff = None, float("inf")
        total_gpu_num = args.node_num * args.gpu_per_node

        def update_opt_explorer_num(trainer_gpu_num, opt_explorer_num, opt_ratio_diff):
            if batch_size % trainer_gpu_num != 0:
                return opt_explorer_num, opt_ratio_diff
            explorer_gpu_num = total_gpu_num - trainer_gpu_num
            if explorer_gpu_num % tensor_parallel_size != 0:
                return opt_explorer_num, opt_ratio_diff
            explorer_num = explorer_gpu_num // tensor_parallel_size
            ratio = explorer_num / trainer_gpu_num
            if opt_ratio_diff > abs(ratio - args.explorer_trainer_ratio):
                return explorer_num, abs(ratio - args.explorer_trainer_ratio)
            return opt_explorer_num, opt_ratio_diff

        if args.node_num == 1:  # single node
            for trainer_gpu_num in range(1, args.gpu_per_node):
                opt_explorer_num, opt_ratio_diff = update_opt_explorer_num(
                    trainer_gpu_num, opt_explorer_num, opt_ratio_diff
                )
        else:  # multi node
            assert (
                args.gpu_per_node % tensor_parallel_size == 0
            ), "Please adjust the value of `tensor_parallel_size` so that it is a divisor of `gpu_per_node`."
            for trainer_node_num in range(1, args.node_num):
                trainer_gpu_num = args.gpu_per_node * trainer_node_num
                opt_explorer_num, opt_ratio_diff = update_opt_explorer_num(
                    trainer_gpu_num, opt_explorer_num, opt_ratio_diff
                )
        assert (
            opt_explorer_num is not None
        ), "Cannot find a suitable explorer number. Please check the value of `train_batch_size`."
        config["explorer"]["rollout_model"]["engine_num"] = opt_explorer_num


def check_taskset_path(dataset_name: str, taskset_path: str) -> str:
    """Ensures the taskset path exists for the given dataset; generates it if necessary.

    This function checks whether `taskset_path` exists. If not,
    it uses a corresponding data generation script (e.g., gen_countdown_data.py) to create
    the dataset at the default or provided location. The generator scripts are expected
    to be located in the 'scripts/' subdirectory relative to this file.

    Args:
        dataset_name: Name of the dataset (e.g., "countdown", "guru").
            Must be one of the supported datasets defined in `dataset_script_map`.
        taskset_path: Path to the dataset.

    Returns:
        str: The resolved path to the dataset.

    Raises:
        ValueError: If the `dataset_name` is not supported.
        FileNotFoundError: If the corresponding generator script does not exist.
        ImportError: If the generator module fails to load.
        AttributeError: If the loaded module does not define 'DEFAULT_DATA_PATH'.
        subprocess.CalledProcessError: If the generation script fails (due to check=True).

    Side Effects:
        - May create directories and files on disk via the external generation script.
        - Executes a subprocess to run the dataset generation script.

    Examples:
        For dataset_name='guru_math' and taskset_path=None, this function will runs the
        following command and generate the guru_math dataset to default location
        (DEFAULT_DATA_PATH in scripts/gen_guru_math_data.py):

        ```bash
        python scripts/gen_guru_math_data.py --local_dir DEFAULT_DATA_PATH
        ```
    """
    if taskset_path:
        if os.path.exists(taskset_path):
            return taskset_path
        if dataset_name == "gsm8k" and taskset_path == "openai/gsm8k":
            return taskset_path

    base_dir = os.path.dirname(__file__)
    frozenlake_data_script_path = os.path.abspath(
        os.path.join(
            base_dir,
            "..",
            "examples",
            "grpo_frozen_lake",
            "get_frozen_lake_data.py",
        )
    )
    dataset_script_map = {
        "countdown": "gen_countdown_data.py",
        "guru_math": "gen_guru_math_data.py",
        "alfworld": "get_alfworld_full_data.py",
        "frozenlake": frozenlake_data_script_path,
    }
    if dataset_name not in dataset_script_map:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. Please specify a valid taskset path."
        )

    script_filename = dataset_script_map[dataset_name]
    script_module_name = script_filename[:-3]  # remove .py

    script_file_path = os.path.join(base_dir, "scripts", script_filename)
    if not os.path.exists(script_file_path):
        raise FileNotFoundError(f"Generator script not found: {script_file_path}")

    spec = importlib.util.spec_from_file_location(script_module_name, script_file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for module: {script_module_name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if taskset_path is None:
        if not hasattr(module, "DEFAULT_DATA_PATH"):
            raise AttributeError(f"{script_filename} is missing 'DEFAULT_DATA_PATH'")
        taskset_path = module.DEFAULT_DATA_PATH
    taskset_path = os.path.realpath(taskset_path)
    if os.path.exists(taskset_path):
        return taskset_path

    # For frozenlake, check if train.parquet and test.parquet already exist
    if dataset_name == "frozenlake":
        train_path = os.path.join(taskset_path, "train.parquet")
        test_path = os.path.join(taskset_path, "test.parquet")
        if os.path.exists(train_path) and os.path.exists(test_path):
            return taskset_path

    gen_script_path = os.path.join(base_dir, "scripts", script_filename)
    subprocess.run([sys.executable, gen_script_path, "--local_dir", taskset_path], check=True)

    return taskset_path


def prepare_configs(args, rank, current_time):
    base_path = os.path.dirname(os.path.abspath(__file__))

    current_time_str = time.strftime("%Y%m%d-%H%M%S", time.localtime(current_time))
    run_path = os.path.join(base_path, "runs", current_time_str)
    config_path = os.path.join(run_path, "config.yaml")
    if rank == 0:
        os.makedirs(run_path)

        with open(os.path.join(base_path, "config", f"{args.dataset}-template.yaml")) as f:
            config = yaml.safe_load(f)

        config["name"] += f"-{current_time_str}"
        config["checkpoint_root_dir"] = os.path.join(run_path, "checkpoints")
        set_engine_num(config, args)
        config["model"]["model_path"] = (
            args.model_path
            or config["model"]["model_path"]
            or os.environ.get(MODEL_PATH_ENV_VAR, "Qwen/Qwen2.5-1.5B-Instruct")
        )
        if ALGORITHM_TYPE.get(config["algorithm"]["algorithm_type"]).use_critic:
            config["model"]["critic_model_path"] = (
                args.critic_model_path
                or config["model"].get("critic_model_path")
                or config["model"]["model_path"]
            )
            if args.critic_lr:
                config["trainer"]["trainer_config"]["critic"]["optim"]["lr"] = args.critic_lr
        if args.dataset == "alfworld":
            print(
                "Warning: The current benchmark script of ALFWorld only supports GRPO; the SFT stage will be supported soon."
            )
        taskset_config = config["buffer"]["explorer_input"]["taskset"]
        taskset_config["path"] = check_taskset_path(
            args.dataset,
            args.taskset_path or os.environ.get("TASKSET_PATH") or taskset_config["path"],
        )
        eval_taskset_config = config["buffer"]["explorer_input"]["eval_tasksets"]
        if len(eval_taskset_config) > 0:
            # TODO: support separately set path for eval taskset
            for eval_taskset_config in eval_taskset_config:
                eval_taskset_config["path"] = taskset_config["path"]
        if args.lr:
            config["algorithm"]["optimizer"]["lr"] = args.lr
        if args.sync_interval:
            config["synchronizer"]["sync_interval"] = args.sync_interval
        if args.sync_offset:
            config["synchronizer"]["sync_offset"] = args.sync_offset
        if args.sync_style:
            config["synchronizer"]["sync_style"] = args.sync_style
        if args.trainer_strategy:
            config["trainer"]["trainer_strategy"] = args.trainer_strategy

        with open(config_path, "w") as f:
            yaml.dump(config, f, allow_unicode=True, sort_keys=False)
    return config_path


def setup_dlc():
    envs = get_dlc_env_vars()
    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        world_size=envs["WORLD_SIZE"],
        rank=envs["RANK"],
    )
    if envs["RANK"] == 0:
        current_time = time.time()
        time_tensor = torch.tensor([current_time], device="cpu")
    else:
        time_tensor = torch.tensor([0.0], device="cpu")
    dist.broadcast(time_tensor, src=0)
    return envs["RANK"], time_tensor.item()


def main(args):
    if args.dlc:
        rank, current_time = setup_dlc()
    else:
        rank, current_time = 0, time.time()
    config_path = prepare_configs(args, rank, current_time)
    cmd_list = [
        sys.executable,
        "-m",
        "trinity.cli.launcher",
        "run",
        "--config",
        config_path,
    ]
    if args.dlc:
        dist.barrier()
        dist.destroy_process_group()
        cmd_list.append("--dlc")

    # load plugins
    base_path = os.path.dirname(os.path.abspath(__file__))
    plugin_dir = os.path.join(base_path, "plugins", args.dataset)
    if os.path.exists(plugin_dir):
        cmd_list.append("--plugin-dir")
        cmd_list.append(plugin_dir)

    # run command
    subprocess.run(cmd_list, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        type=str.lower,
        choices=["gsm8k", "countdown", "guru_math", "alfworld", "frozenlake"],
    )
    parser.add_argument(
        "--dlc", action="store_true", help="Specify when running in Aliyun PAI DLC."
    )
    parser.add_argument("--node_num", type=int, default=1, help="Specify the number of nodes.")
    parser.add_argument(
        "--gpu_per_node", type=int, default=8, help="Specify the number of GPUs per node."
    )
    parser.add_argument(
        "--vllm_engine_num", type=int, default=None, help="Specify the number of vLLM engines."
    )
    parser.add_argument(
        "--vllm_tp_size", type=int, default=None, help="Specify the number of vLLM tp size."
    )
    parser.add_argument(
        "--explorer_trainer_ratio",
        type=float,
        default=0.6,
        help="Specify the ratio of explorer engine num to trainer gpu num.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Specify the path to the model checkpoint.",
    )
    parser.add_argument(
        "--critic_model_path",
        type=str,
        default=None,
        help="Specify the path to the critic model checkpoint.",
    )
    parser.add_argument(
        "--taskset_path", type=str, default=None, help="Specify the path to the taskset."
    )
    parser.add_argument(
        "--lr", type=float, default=None, help="Specify the learning rate for actor model."
    )
    parser.add_argument(
        "--critic_lr", type=float, default=None, help="Specify the learning rate for critic model."
    )
    parser.add_argument(
        "--sync_interval", type=int, default=None, help="Specify the sync interval."
    )
    parser.add_argument("--sync_offset", type=int, default=None, help="Specify the sync offset.")
    parser.add_argument(
        "--sync_style",
        type=str,
        default=None,
        choices=[sync_style.value for sync_style in SyncStyle],
    )
    parser.add_argument(
        "--trainer_strategy",
        type=str,
        default=None,
        choices=["fsdp", "fsdp2", "megatron"],
        help="Specify the trainer strategy.",
    )
    args = parser.parse_args()
    main(args)
