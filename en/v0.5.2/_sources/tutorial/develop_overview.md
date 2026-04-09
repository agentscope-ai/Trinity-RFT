# Developer Guide

Trinity-RFT divides the RL training process into three modules: **Explorer**, **Trainer**, and **Buffer**.

Explorer provides the `Workflow` interface to customize agent-environment interaction, Trainer provides the `Algorithm` interface to customize RL algorithms, and Buffer provides the `Operator` interface to customize data processing operators.

The table below lists the main functions of each extension interface, its target users, and the corresponding development tutorials. Developers can refer to the respective module development tutorials and extend Trinity-RFT based on their needs.

| Extension Interface | Target Users      | Main Functions                            | Tutorial Link              |
|---------------------|-------------------|------------------------------------------|----------------------------|
| `Workflow`          | Agent Application Developers | Enhance agent's ability to complete tasks in a specified environment | [🔗](./develop_workflow.md) |
| `Algorithm`         | RL Algorithm Researchers | Design new RL algorithms                 | [🔗](./develop_algorithm.md) |
| `Operator`          | Data Engineers    | Design new data cleaning and augmentation strategies | [🔗](./develop_operator.md) |
| `Selector`          | Data Engineers    | Design new task selection strategies           | [🔗](./develop_selector.md) |

```{tip}
Trinity-RFT provides a modular development approach, allowing you to flexibly add custom modules without modifying the framework code.
You can place your module code in the `trinity/plugins` directory. Trinity-RFT will automatically load all Python files in that directory at runtime and register the custom modules within them.
Trinity-RFT also supports specifying other directories at runtime by setting the `--plugin-dir` option, for example: `trinity run --config <config_file> --plugin-dir <your_plugin_dir>`.
Alternatively, you can use the relative path to the custom module in the YAML configuration file, for example: `default_workflow_type: 'examples.agentscope_frozenlake.workflow.FrozenLakeWorkflow'`.
```

For modules you plan to contribute to Trinity-RFT, please follow these steps:

1. Implement your code in the appropriate directory, such as `trinity/common/workflows` for `Workflow`, `trinity/algorithm` for `Algorithm`, and `trinity/buffer/operators` for `Operator`.

2. Register your module in the corresponding mapping dictionary in the `__init__.py` file of the directory.
   For example, if you want to register a new workflow class `ExampleWorkflow`, you need to modify the `default_mapping` dictionary of `WORKFLOWS` in the `trinity/common/workflows/__init__.py` file:
   ```python
   WORKFLOWS: Registry = Registry(
       "workflows",
       default_mapping={
           "example_workflow": "trinity.common.workflows.workflow.ExampleWorkflow",
       },
   )
   ```

3. Add tests for your module in the `tests` directory, following the naming conventions and structure of existing tests.

4. Before submitting your code, ensure it passes the code style check by running `pre-commit run --all-files`.

5. Submit a Pull Request to the Trinity-RFT repository, providing a detailed description of your module's functionality and purpose.
