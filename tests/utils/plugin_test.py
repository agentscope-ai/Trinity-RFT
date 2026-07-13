import os
import shutil
import unittest
from pathlib import Path
from typing import Type
from unittest.mock import MagicMock

import ray
from parameterized import parameterized

from tests.tools import TensorBoardParser, get_checkpoint_path, get_template_config
from trinity.common.config import Config
from trinity.common.constants import PLUGIN_DIRS_ENV_VAR
from trinity.common.workflows import WORKFLOWS, Task, Workflow
from trinity.utils.monitor import MONITOR
from trinity.utils.plugin_loader import load_plugins


class PluginActor:
    def __init__(
        self,
        config: Config,
        enable_load_plugins: bool = True,
        enable_monitor: bool = True,
        enable_workflow: bool = True,
    ):
        if enable_load_plugins:
            load_plugins()
        self.config = config
        if enable_monitor:
            self.monitor = MONITOR.get("my_monitor")(
                project=self.config.project,
                group=self.config.group,
                name=self.config.name,
                role=self.config.explorer.name,
                config=config,
            )
        else:
            self.monitor = None
        if enable_workflow:
            workflow = WORKFLOWS.get("my_workflow")
            assert workflow is not None, "Workflow 'my_workflow' not found in registry"

    def run(self, workflow_cls=Type[Workflow]):
        if self.monitor:
            self.monitor.log({"rollout": 2}, step=1, commit=True)
        return workflow_cls(task=Task(), model=MagicMock()).run()


class TestPluginLoader(unittest.TestCase):
    PLUGIN_DIR_PARAMS = [
        (str(Path(__file__).resolve().parent / "plugins"),),
        (os.path.join("tests", "utils", "plugins"),),
    ]

    @parameterized.expand(PLUGIN_DIR_PARAMS)
    def test_load_plugins_local(self, plugin_dir):
        if os.path.isabs(plugin_dir):
            with self.assertRaises(ValueError):
                my_workflow_cls = WORKFLOWS.get("my_workflow")
        os.environ[PLUGIN_DIRS_ENV_VAR] = plugin_dir
        try:
            load_plugins()
        except KeyError:
            # already registered in next test
            pass
        my_workflow_cls = WORKFLOWS.get("my_workflow")
        self.assertIsNotNone(my_workflow_cls)
        my_plugin = my_workflow_cls(task=Task(), model=MagicMock(), auxiliary_models=None)
        self.assertTrue(my_plugin.__module__.startswith("trinity.plugins"))
        res = my_plugin.run()
        self.assertEqual(res[0], "Hello world")
        self.assertEqual(res[1], "Hi")

    @parameterized.expand(PLUGIN_DIR_PARAMS)
    def test_load_plugins_remote(self, plugin_dir):
        os.environ[PLUGIN_DIRS_ENV_VAR] = plugin_dir
        try:
            load_plugins()
        except KeyError:
            # already registered in previous test
            pass
        config = self.config
        ray.init(
            ignore_reinit_error=True,
            runtime_env={"env_vars": {PLUGIN_DIRS_ENV_VAR: plugin_dir}},
        )
        # disable plugin and use custom class from registry
        remote_plugin = ray.remote(PluginActor).remote(config, enable_load_plugins=False)
        with self.assertRaises(ray.exceptions.ActorDiedError):
            # During initialization, enable_workflow=True (default) will try to get "my_workflow"
            ray.get(remote_plugin.__ray_ready__.remote())

        # enable plugin
        remote_plugin = ray.remote(PluginActor).remote(config)
        my_workflow_cls = WORKFLOWS.get("my_workflow")
        remote_res = ray.get(remote_plugin.run.remote(my_workflow_cls))
        self.assertEqual(remote_res[0], "Hello world")
        self.assertEqual(remote_res[1], "Hi")

        # test custom monitor
        parser = TensorBoardParser(os.path.join(config.monitor.cache_dir, "tensorboard"))
        rollout_cnt = parser.metric_values("rollout")
        self.assertEqual(rollout_cnt, [2])

    @parameterized.expand(PLUGIN_DIR_PARAMS)
    def test_passing_custom_class(self, plugin_dir):
        # disable plugin and pass custom class directly
        os.environ[PLUGIN_DIRS_ENV_VAR] = plugin_dir
        try:
            load_plugins()
        except KeyError:
            # already registered in previous test
            pass
        my_workflow_cls = WORKFLOWS.get("my_workflow")
        remote_plugin = ray.remote(PluginActor).remote(
            self.config, enable_load_plugins=False, enable_monitor=False, enable_workflow=False
        )
        remote_res = ray.get(remote_plugin.run.remote(my_workflow_cls))
        self.assertEqual(remote_res[0], "Hello world")
        ray.shutdown(_exiting_interpreter=True)

    def setUp(self):
        self.config = get_template_config()
        self.config.checkpoint_root_dir = get_checkpoint_path()
        self.config.check_and_update()
        shutil.rmtree(self.config.monitor.cache_dir, ignore_errors=True)
