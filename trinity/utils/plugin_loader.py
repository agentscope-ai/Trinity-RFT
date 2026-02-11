"""Load modules from custom directory"""

import importlib
import os
import shutil
import sys
from pathlib import Path
from typing import List, Union

from trinity.common.constants import PLUGIN_DIRS_ENV_VAR
from trinity.utils.log import get_logger


def load_plugins() -> None:
    """
    Load plugin modules from the default plugin directory or directories specified in the environment variable.
    If the environment variable `PLUGIN_DIRS_ENV_VAR` is not set, it defaults to `trinity/plugins`.
    """
    plugin_dirs = os.environ.get(PLUGIN_DIRS_ENV_VAR, "").split(os.pathsep)
    if not plugin_dirs or plugin_dirs == [""]:
        plugin_dirs = [str(Path(__file__).parent.parent / "plugins")]

    load_plugin_from_dirs(plugin_dirs)


def load_plugin_from_dirs(plugin_dirs: Union[str, List[str]]) -> None:
    """
    Load plugin modules from a directory.
    """
    logger = get_logger(__name__)
    if not isinstance(plugin_dirs, list):
        plugin_dirs = [plugin_dirs]
    plugin_dirs = set(plugin_dirs)
    for plugin_dir in plugin_dirs:
        if not os.path.exists(plugin_dir):
            logger.error(f"plugin-dir [{plugin_dir}] does not exist.")
            continue
        if not os.path.isdir(plugin_dir):
            logger.error(f"plugin-dir [{plugin_dir}] is not a directory.")
            continue

        for file in Path(plugin_dir).glob("*.py"):
            if file.name.startswith("__"):
                continue
            logger.info(f"Loading plugin modules from [{file}]...")
            # load modules from file
            try:
                load_from_file(str(file))
            except Exception as e:
                logger.warning(f"Failed to load plugin module from [{file}]: {e}")


def load_from_file(file_path: str):
    """
    Load modules from a Python file

    Args:
        file_path (`str`): The python file path.

    Returns:
        `Any`: The loaded module.
    """
    module_name = os.path.splitext(os.path.basename(file_path))[0]

    full_module_name = f"trinity.plugins.{module_name}"

    spec = importlib.util.spec_from_file_location(full_module_name, file_path)
    if spec is None:
        raise ImportError(f"Cannot load module from {file_path}")

    module = importlib.util.module_from_spec(spec)

    module.__package__ = "trinity.plugins"

    spec.loader.exec_module(module)

    if full_module_name in sys.modules:
        raise ImportError(f"Module {module_name} already exists.")
    sys.modules[full_module_name] = module
    try:
        shutil.copy2(file_path, Path(__file__).parent.parent / "plugins")
    except shutil.SameFileError:
        pass
    return module
