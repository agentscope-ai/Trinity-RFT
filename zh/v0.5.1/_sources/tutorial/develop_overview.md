# 开发者指南

Trinity-RFT 将 RL 训练过程拆分为了三个模块：**Explorer**、**Trainer** 和 **Buffer**。

其中 Explorer 提供了能够自定义智能体与环境交互的 `Workflow` 接口，Trainer 提供了能够自定义 RL 算法的 `Algorithm` 接口，Buffer 提供了能够自定义数据处理算子的 `Operator` 接口。

下表列出了各扩展接口的主要目标用户、功能以及对应的开发教程。开发者可以参考对应的模块开发教程，并根据自身需求对 Trinity-RFT 进行扩展。

| 扩展接口      | 目标用户        | 主要功能                            |  教程链接          |
|--------------|----------------|-----------------------------------|------------------|
| `Workflow`   | 智能体应用开发者 | 提升 Agent 在指定环境中完成任务的能力  | [🔗](./develop_workflow.md) |
| `Algorithm`  | RL 算法研究者   | 设计新的 RL 算法                    | [🔗](./develop_algorithm.md) |
| `Operator`   | 数据工程师      | 设计新的数据清洗、增强策略            | [🔗](./develop_operator.md) |
| `Selector`   | 数据工程师      | 设计新的数据选择策略                  | [🔗](./develop_selector.md) |

```{tip}
Trinity-RFT 提供了插件化的开发方式，可以在不修改框架代码的前提下，灵活地添加自定义模块。
开发者可以将自己编写的模块代码放在 `trinity/plugins` 目录下。Trinity-RFT 会在运行时自动加载该目录下的所有 Python 文件，并注册其中的自定义模块。
Trinity-RFT 也支持在运行时通过设置 `--plugin-dir` 选项来指定其他目录，例如：`trinity run --config <config_file> --plugin-dir <your_plugin_dir>`。
另外，你也可以使用相对路径来在 YAML 配置文件中指定自定义模块，例如：`default_workflow_type: 'examples.agentscope_frozenlake.workflow.FrozenLakeWorkflow'`。
```

对于准备向 Trinity-RFT 提交的模块，请遵循以下步骤：

1. 在适当目录中实现你的代码，例如 `trinity/common/workflows` 用于 `Workflow`，`trinity/algorithm` 用于 `Algorithm`，`trinity/buffer/operators` 用于 `Operator`。

2. 在目录对应的 `__init__.py` 文件中的 `default_mapping` 字典中注册你的模块。例如，对于新的 `ExampleWorkflow` 类，你需要在 `trinity/common/workflows/__init__.py` 文件中的 `WORKFLOWS` 中添加你的模块：
   ```python
   WORKFLOWS: Registry = Registry(
       "workflows",
       default_mapping={
           "example_workflow": "trinity.common.workflows.workflow.ExampleWorkflow",
       },
   )
   ```

3. 在 `tests` 目录中为你的模块添加测试，遵循现有测试的命名约定和结构。

4. 提交代码前，确保通过 `pre-commit run --all-files` 完成代码风格检查。

5. 向 Trinity-RFT 仓库提交 Pull Request，在描述中详细说明你的模块功能和用途。
