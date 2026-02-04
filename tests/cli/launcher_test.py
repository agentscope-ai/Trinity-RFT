import multiprocessing
import os
import shutil
import sys
import time
import unittest
from unittest import mock
from unittest.mock import MagicMock

from tests.tools import (
    get_checkpoint_path,
    get_model_path,
    get_template_config,
    get_unittest_dataset_config,
)
from trinity.cli import launcher
from trinity.common.config import (
    AlgorithmConfig,
    BufferConfig,
    StageConfig,
    TrainerInput,
)
from trinity.common.constants import (
    LOG_DIR_ENV_VAR,
    LOG_LEVEL_ENV_VAR,
    LOG_NODE_IP_ENV_VAR,
)
from trinity.common.models import get_debug_explorer_model


class TestLauncherMain(unittest.TestCase):
    def setUp(self):
        if multiprocessing.get_start_method(allow_none=True) != "spawn":
            multiprocessing.set_start_method("spawn", force=True)
        self._orig_argv = sys.argv.copy()
        self.config = get_template_config()
        self.config.checkpoint_root_dir = get_checkpoint_path()
        self.config.model.model_path = get_model_path()
        self.config.check_and_update()
        shutil.rmtree(self.config.checkpoint_job_dir, ignore_errors=True)

    def tearDown(self):
        sys.argv = self._orig_argv
        shutil.rmtree(self.config.checkpoint_job_dir, ignore_errors=True)

    @mock.patch("trinity.cli.launcher.serve")
    @mock.patch("trinity.cli.launcher.explore")
    @mock.patch("trinity.cli.launcher.train")
    @mock.patch("trinity.cli.launcher.both")
    @mock.patch("trinity.cli.launcher.bench")
    @mock.patch("trinity.cli.launcher.load_config")
    def test_main_run_command(
        self, mock_load, mock_bench, mock_both, mock_train, mock_explore, mock_serve
    ):
        config = get_template_config()
        mapping = {
            "explore": mock_explore,
            "train": mock_train,
            "both": mock_both,
            "bench": mock_bench,
            "serve": mock_serve,
        }
        with mock.patch.dict(
            launcher.MODE_MAP,
            {
                "explore": mock_explore,
                "train": mock_train,
                "both": mock_both,
                "bench": mock_bench,
                "serve": mock_serve,
            },
        ):
            for mode in ["explore", "train", "both", "bench", "serve"]:
                config.mode = mode
                mock_load.return_value = config
                with mock.patch(
                    "argparse.ArgumentParser.parse_args",
                    return_value=mock.Mock(
                        command="run", config="dummy.yaml", dlc=False, plugin_dir=None
                    ),
                ):
                    launcher.main()
                mock_load.assert_called_once_with("dummy.yaml")
                mapping[mode].assert_called_once_with(config)
                mock_load.reset_mock()
                mapping[mode].reset_mock()

    @mock.patch("trinity.cli.launcher.stop_ray_cluster")
    @mock.patch("trinity.cli.launcher.setup_ray_cluster")
    @mock.patch("trinity.cli.launcher.both")
    @mock.patch("trinity.cli.launcher.load_config")
    @mock.patch("ray.init")
    def test_main_run_in_dlc(self, mock_init, mock_load, mock_both, mock_setup, mock_stop):
        config = get_template_config()
        namespace = f"{config.project}-{config.name}"
        config.mode = "both"
        config.log.level = "WARNING"
        config.log.group_by_node = True
        mock_setup.return_value = "auto"
        mock_load.return_value = config
        with mock.patch.dict(
            launcher.MODE_MAP,
            {
                "both": mock_both,
            },
        ):
            with mock.patch(
                "argparse.ArgumentParser.parse_args",
                return_value=mock.Mock(
                    command="run", config="dummy.yaml", dlc=True, plugin_dir="/path/to/plugins"
                ),
            ):
                launcher.main()
            mock_init.assert_called_once()
            mock_init.assert_called_once_with(
                address="auto",
                ignore_reinit_error=True,
                namespace=config.ray_namespace,
                runtime_env={
                    "env_vars": {
                        launcher.PLUGIN_DIRS_ENV_VAR: "/path/to/plugins",
                        LOG_DIR_ENV_VAR: config.log.save_dir,
                        LOG_LEVEL_ENV_VAR: config.log.level,
                        LOG_NODE_IP_ENV_VAR: "1",
                    }
                },
            )
            mock_load.assert_called_once_with("dummy.yaml")
            mock_both.assert_called_once_with(config)
            mock_setup.assert_called_once_with(
                namespace=namespace,
            )
            mock_stop.assert_called_once_with(
                namespace=namespace,
            )

    @mock.patch("trinity.cli.launcher.studio")
    def test_main_studio_command(self, mock_studio):
        with mock.patch(
            "argparse.ArgumentParser.parse_args",
            return_value=mock.Mock(command="studio", port=9999),
        ):
            launcher.main()
        mock_studio.assert_called_once_with(9999)

    @mock.patch("trinity.trainer.verl.utils.get_latest_hf_checkpoint_path")
    @mock.patch("trinity.cli.launcher.both")
    @mock.patch("trinity.cli.launcher.train")
    @mock.patch("trinity.cli.launcher.load_config")
    @mock.patch("ray.shutdown")
    @mock.patch("ray.init")
    def test_multi_stage_run(
        self,
        mock_init: MagicMock,
        mock_shutdown: MagicMock,
        mock_load: MagicMock,
        mock_train: MagicMock,
        mock_both: MagicMock,
        mock_checkpoint_path: MagicMock,
    ):
        config = get_template_config()
        config.ray_namespace = ""
        config.checkpoint_root_dir = get_checkpoint_path()
        config.model.model_path = get_model_path()
        config.stages = [
            StageConfig(
                mode="train",
                stage_name="sft_warmup",
                algorithm=AlgorithmConfig(
                    algorithm_type="sft",
                ),
                buffer=BufferConfig(
                    train_batch_size=32,
                    total_steps=100,
                    trainer_input=TrainerInput(
                        experience_buffer=get_unittest_dataset_config("sft_for_gsm8k")
                    ),
                ),
            ),
            StageConfig(
                mode="both",
                stage_name="grpo",
                algorithm=AlgorithmConfig(
                    algorithm_type="grpo",
                ),
            ),
        ]
        mock_load.return_value = config
        mock_checkpoint_path.return_value = "/path/to/hf/checkpoint"
        with mock.patch.dict(
            launcher.MODE_MAP,
            {
                "train": mock_train,
                "both": mock_both,
            },
        ):
            with mock.patch(
                "argparse.ArgumentParser.parse_args",
                return_value=mock.Mock(
                    command="run", config="dummy.yaml", dlc=False, plugin_dir="/path/to/plugins"
                ),
            ):
                launcher.main()
            self.assertEqual(mock_init.call_count, 2)
            self.assertEqual(mock_shutdown.call_count, 2)
            mock_train.assert_called_once()
            mock_both.assert_called_once()
            expected_calls = [
                mock.call(
                    address="auto",
                    ignore_reinit_error=True,
                    namespace=f"{config.project}/{config.name}/sft_warmup",
                    runtime_env={
                        "env_vars": {
                            launcher.PLUGIN_DIRS_ENV_VAR: "/path/to/plugins",
                            LOG_DIR_ENV_VAR: os.path.join(
                                config.checkpoint_root_dir,
                                config.project,
                                f"{config.name}/sft_warmup",
                                "log",
                            ),
                            LOG_LEVEL_ENV_VAR: config.log.level,
                            LOG_NODE_IP_ENV_VAR: "0",
                        }
                    },
                ),
                mock.call(
                    address="auto",
                    ignore_reinit_error=True,
                    namespace=f"{config.project}/{config.name}/grpo",
                    runtime_env={
                        "env_vars": {
                            launcher.PLUGIN_DIRS_ENV_VAR: "/path/to/plugins",
                            LOG_DIR_ENV_VAR: os.path.join(
                                config.checkpoint_root_dir,
                                config.project,
                                f"{config.name}/grpo",
                                "log",
                            ),
                            LOG_LEVEL_ENV_VAR: config.log.level,
                            LOG_NODE_IP_ENV_VAR: "0",
                        }
                    },
                ),
            ]
            mock_init.assert_has_calls(expected_calls)
            self.assertEqual(mock_checkpoint_path.call_count, 2)
            self.assertEqual(mock_train.call_args[0][0].model.model_path, config.model.model_path)
            self.assertEqual(mock_both.call_args[0][0].model.model_path, "/path/to/hf/checkpoint")
            self.assertEqual(
                mock_both.call_args[0][0].trainer.trainer_config.actor_rollout_ref.model.path,
                "/path/to/hf/checkpoint",
            )

    @mock.patch("trinity.cli.launcher.load_config")
    def test_debug_mode(self, mock_load):
        process = multiprocessing.Process(target=debug_inference_model_process)
        try:
            process.start()
            time.sleep(15)  # wait for the model to be created
            for _ in range(10):
                try:
                    get_debug_explorer_model(self.config)
                    break
                except Exception:
                    time.sleep(3)
            output_file = os.path.join(self.config.checkpoint_job_dir, "debug.html")
            output_dir = os.path.join(self.config.checkpoint_job_dir, "debug_output")
            self.config.buffer.explorer_input.tasksets = [get_unittest_dataset_config("gsm8k")]
            mock_load.return_value = self.config
            with mock.patch(
                "argparse.ArgumentParser.parse_args",
                return_value=mock.Mock(
                    command="debug",
                    config="dummy.yaml",
                    module="workflow",
                    enable_profiling=True,
                    disable_overwrite=False,
                    output_dir=output_dir,
                    output_file=output_file,
                    plugin_dir="",
                ),
            ):
                launcher.main()

            self.assertFalse(os.path.exists(output_file))
            self.assertTrue(os.path.exists(output_dir))
            self.assertTrue(os.path.exists(os.path.join(output_dir, "profiling.html")))
            self.assertTrue(os.path.exists(os.path.join(output_dir, "experiences.db")))
            # add a dummy file to test overwrite behavior
            with open(os.path.join(output_dir, "dummy.txt"), "w") as f:
                f.write("not empty")

            with mock.patch(
                "argparse.ArgumentParser.parse_args",
                return_value=mock.Mock(
                    command="debug",
                    config="dummy.yaml",
                    module="workflow",
                    enable_profiling=False,
                    disable_overwrite=False,
                    output_dir=output_dir,
                    output_file=output_file,
                    plugin_dir="",
                ),
            ):
                launcher.main()

            dirs = os.listdir(self.config.checkpoint_job_dir)
            target_output_dir = [d for d in dirs if d.startswith("debug_output_")]
            self.assertEqual(len(target_output_dir), 0)

            with mock.patch(
                "argparse.ArgumentParser.parse_args",
                return_value=mock.Mock(
                    command="debug",
                    config="dummy.yaml",
                    module="workflow",
                    enable_profiling=False,
                    disable_overwrite=True,
                    output_dir=output_dir,
                    output_file=output_file,
                    plugin_dir="",
                ),
            ):
                launcher.main()

            self.assertFalse(os.path.exists(output_file))
            # test the original files are not overwritten
            self.assertTrue(os.path.exists(output_dir))
            self.assertTrue(os.path.exists(os.path.join(output_dir, "dummy.txt")))
            dirs = os.listdir(self.config.checkpoint_job_dir)
            target_output_dir = [d for d in dirs if d.startswith("debug_output_")]
            self.assertEqual(len(target_output_dir), 1)
            self.assertFalse(
                os.path.exists(
                    os.path.join(
                        self.config.checkpoint_job_dir, target_output_dir[0], "profiling.html"
                    )
                )
            )
            self.assertTrue(
                os.path.exists(
                    os.path.join(
                        self.config.checkpoint_job_dir, target_output_dir[0], "experiences.db"
                    )
                )
            )
        finally:
            process.join(timeout=10)
            process.terminate()


def debug_inference_model_process():
    config = get_template_config()
    config.checkpoint_root_dir = get_checkpoint_path()
    config.model.model_path = get_model_path()
    config.check_and_update()
    with mock.patch("trinity.cli.launcher.load_config", return_value=config):
        with mock.patch(
            "argparse.ArgumentParser.parse_args",
            return_value=mock.Mock(
                command="debug",
                config="dummy.yaml",
                module="inference_model",
                plugin_dir=None,
                output_file=None,
            ),
        ):
            launcher.main()
