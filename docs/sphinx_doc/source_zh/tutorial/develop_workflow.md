(Workflows)=
## Workflow 开发指南

在 Trinity-RFT 中，工作流（Workflow）是定义 Agent 与 Environment 之间交互的核心组件。
一个合格的工作流需要使用被训练模型完成指定任务，并从环境中获取反馈信息（奖励）。本节将会介绍如何开发一个新的工作流。

---

### 步骤 0：基本概念

在开发之前，理解以下几个核心概念非常重要：

```{mermaid}
flowchart LR
    A([Task]) --> C[Workflow]
    C -- "调用 OpenAI API" --> B([Rollout Model])
    B -- "自动 recording" --> D([Experience])
    C -- "update_reward" --> D
```

- **任务（Task）** ({class}`trinity.common.workflows.Task`)：结构化的数据实例，包含了工作流一次运行所需的各种信息。一般情况下由训练数据集提供，数据集中的每个样本都会被转化为一个 `Task` 实例。`Task` 的内容根据任务类型而异：
  - **数学问题**：包含问题和答案。
  - **编程场景**：包含题目的描述、测试用例、运行环境等复杂信息。

- **模型（Rollout Model）** ({class}`trinity.common.models.model.ModelWrapper`)：被训练的模型。工作流通过模型暴露的 `base_url` 和 `api_key` 自行创建 OpenAI 客户端来调用模型推理接口；模型在响应的同时会**自动记录**生成过程并转化为可用于训练的 `Experience`，工作流无需手动构造。

- **工作流（Workflow）** ({class}`trinity.common.workflows.WorkflowBase`)：定义了 Agent 与 Environment 的交互流程。`Workflow` 通过 `Task` 中提供的信息初始化自身，并借助 Rollout Model 执行其中定义好的交互流程。与常规 Agent 应用不同的是，工作流内部还需要计算奖励信号（reward）以指导训练过程，并通过 `update_reward` 方法将奖励回填到模型自动记录的 `Experience` 上。

- **经验（Experience）** ({class}`trinity.common.experience.Experience`)：训练所需的数据单元。`Experience` 会由 Rollout Model 在推理过程中自动记录产生，其数量与内部数据格式取决于所使用的训练算法。例如，对于常见的 PPO/GRPO 算法，`Experience` 包含 token ID 列表、动作掩码（标识哪些 token 是由 LLM 生成的）、每个 token 的对数概率（logprobs）、奖励信号（reward）等。工作流不需要、也不应该手动构造 `Experience` 对象。

---

### 步骤 1：准备任务数据集

任务数据集通过 YAML 配置文件中的 `buffer.explorer_input.taskset` 配置项加载。
为处理 `Task` 内容的差异，Trinity-RFT 提供了一个统一的 `Task` 接口，包含以下字段：

- **`workflow`** (`str`)：你的工作流类的注册名称。你可以在 YAML 配置文件的 `buffer.explorer_input.taskset.default_workflow_type` 中指定。
- **`raw_task`** (`Dict`)：原始数据的记录，以 `Dict` 格式存储。对于高度定制化的工作流，你可以直接使用 `raw_task` 初始化 `Workflow` 实例，而不依赖后续的字段。

下面的字段都是可选字段，一般情况下无需设置：
- **`reward_fn`** (`Optional[str]`)：你的奖励函数的注册名称。你可以在 `buffer.explorer_input.taskset.default_reward_fn_type` 中指定。注意某些工作流已内置奖励计算；此时可省略该字段。
- **`format_args`** ({class}`trinity.common.config.FormatConfig`)：便于构造 `Workflow` 实例的参数。例如，`prompt_key` 和 `response_key` 可用于从 `raw_task` 中提取 prompt 和 response。这些设置来自 YAML 配置文件，可在 `buffer.explorer_input.task_set.format` 中设置。
- **`rollout_args`** ({class}`trinity.common.config.GenerationConfig`)：控制 rollout 过程的参数，如 `temperature`。该字段也来自 YAML 配置文件，可在 `buffer.explorer_input.task_set.rollout_args` 中设置。
- **`workflow_args`** (`Dict`)：用于构造 `Workflow` 实例的参数字典。相比 `format_args` 和 `rollout_args` 更灵活。该字段也来自 YAML 配置文件，可在 `buffer.explorer_input.task_set.workflow_args` 中设置。通常无需设置此字段。

```{tip}
`workflow`、`workflow_args` 和 `raw_task` 提供了不同级别的自定义能力。

- `workflow` 为使用相同工作流的所有任务提供全局设置。（全局级别）
- `workflow_args` 可为每个任务数据集设置，允许使用相同工作流的不同任务数据集表现出不同行为。（数据集级别）
- `raw_task` 提供对每个任务行为的自定义能力，最为灵活。（数据样本级别）
```

在数学问题场景中，`Task` 数据集可以是一个 `jsonl` 文件，每行包含带有 `question` 和 `answer` 字段的 JSON，分别表示问题描述和标准答案。例如：

```json
{"question": "1+1=", "answer": "2"}
{"question": "2+2=", "answer": "4"}
...
```

配置示例片段：

```yaml
# some config
buffer:
  explorer_input:
    taskset:
      default_workflow_type: "math_workflow"
      path: ${oc.env:TRINITY_TASKSET_PATH}
      format:
        prompt_key: "question"
        response_key: "answer"
      rollout_args:
        temperature: 1.0
      # some other configs
```

在此示例中，每个任务对象的 `raw_task` 是一个包含两个键（`question` 和 `answer`）的 `Dict`。`MathWorkflow` 使用 `prompt_key` 和 `response_key` 从 `raw_task` 中提取问题和答案，并使用 `rollout_args` 生成响应。

---

### 步骤 2：实现工作流

要实现一个新的工作流你需要继承 `WorkflowWithRecording` 基类：

```python
class WorkflowWithRecording(WorkflowBase):

    def __init__(
        self,
        *,
        task: Task,
        model: ModelWrapper,
        auxiliary_models: Optional[List[ModelWrapper]] = None,
    ):
        """初始化工作流"""

    async def run_async(self) -> Metrics:
        """运行工作流并返回一个 Metric 字典。"""
        # 你需要实现该方法

    @property
    def base_url(self) -> str:
        """返回 rollout 模型的 base_url。"""

    @property
    def api_key(self) -> str:
        """返回 rollout 模型的 api_key。"""

    @property
    def model_name(self) -> str:
        """返回 rollout 模型的 model_name。"""

    async def update_reward(
        self,
        reward: float,
        info: Optional[Dict] = None,
    ):
        """将 reward 回填到模型自动记录的 Experience 上，同时可选附带额外信息 info。"""

```

#### 初始化你的工作流

`WorkflowWithRecording` 接受以下初始化参数：

- `task`({class}`trinity.common.workflows.Task`)：数据集中的单个任务。
- `model`({class}`trinity.common.models.model.ModelWrapper`)：正在训练的 rollout 模型，你可以直接通过 `WorkflowWithRecording` 的 `base_url`，`api_key` 以及 `model_name` 属性来创建 OpenAI 客户端从而调用模型推理接口。
- `auxiliary_models`(`List[ModelWrapper]`)：辅助模型的 `ModelWrapper` 列表。每个元素同样暴露 `base_url`、`api_key`、`model_name`，可直接用于创建 OpenAI 客户端（详见 [LLM-as-a-judge 支持](#llm-as-a-judge-支持)）。

以下是一个简单工作流的初始化示例。我们在 `__init__` 中使用 `base_url` 和 `api_key` 创建异步 OpenAI 客户端，并取出模型名称：

```python
import openai
from trinity.common.workflows import WorkflowWithRecording

class ExampleWorkflow(WorkflowWithRecording):

    def __init__(self, *, task: Task, model: ModelWrapper, auxiliary_models: List = None):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        self.question = task.raw_task.get("question")
        self.answer = task.raw_task.get("answer")
        self.rollout_args = task.rollout_args
        # 通过 base_url 和 api_key 创建 OpenAI 客户端
        self.client = openai.AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)
```

#### 实现 `run_async` 方法

`run_async` 是工作流的核心方法。它没有输入参数，返回一个 `Metrics` 字典。

工作流的职责是：调用模型完成 agent 任务、计算 reward、通过 `update_reward` 将 reward 回填到模型自动记录的 `Experience` 上，最后返回用于监控的 metric。

以下是一个数学工作流的简单实现。我们先用 OpenAI 客户端生成答案，再计算奖励并回填：

```python
class ExampleWorkflow(WorkflowWithRecording):

    # the __init__ function

    def calculate_reward(self, response: str, truth: str) -> float:
        if response == truth:
            return 1.0
        else:
            return 0.0

    async def run_async(self) -> Metrics:
        # 调用模型生成回复
        resp = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": f"Question:\n{self.question}",
                }
            ],
            temperature=self.rollout_args.temperature,
        )
        response_text = resp.choices[0].message.content
        # 计算 reward 并回填到模型自动记录的 Experience 上
        reward: float = self.calculate_reward(response_text, self.answer)
        await self.update_reward(reward)
        # 返回需要监控的 metric
        return {"example/reward": reward}
```

```{note}
1. rollout 模型会自动记录每次 `chat.completions.create` 调用产生的训练数据并转化为 `Experience`。`update_reward` 会将 reward 精确回填到本次运行产生的 `Experience` 上。
2. 对于包含多轮交互的工作流，`update_reward` 会将 reward 回填到本次运行产生的所有 `Experience` 上。
3. `run_async` 返回的 `Metrics` 字典仅用于运行时监控与日志展示。
```

#### 注册你的工作流

为了让 Trinity-RFT 能够通过配置文件中的名称自动找到你的工作流，你需要将其注册到 `WORKFLOWS` 注册表中。推荐使用装饰器方式注册：

```python
from trinity.common.workflows import WORKFLOWS, WorkflowWithRecording

@WORKFLOWS.register_module(name="example_workflow")
class ExampleWorkflow(WorkflowWithRecording):
    ...
```

也可以直接注册，或在 `trinity/common/workflows/__init__.py` 的 `default_mapping` 中添加一条 `"example_workflow": "path.to.module.ExampleWorkflow"` 映射。

#### 性能调优

对于较为复杂的工作流，每次重新初始化会带来额外计算开销。此时，你可以设置 `can_reset` 类属性并实现 `reset` 方法以避免重复初始化。

注意在 `reset` 方法中必须使用输入的 `task` 覆盖工作流的 `task` 属性，并使用 `task.api_key` 更新模型和客户端的 API Key。

> Trinity-RFT 内部借助 `api_key` 来区分不同任务产生的 Experience，如果不更新 API Key，可能会导致不同任务的 Experience 被错误地归类，导致 reward 回填错误。

以下是一个简单示例：

```python
class ExampleWorkflow(WorkflowWithRecording):
    can_reset: bool = True

    # some code
    # ...

    def reset(self, task: Task):
        self.task = task
        self.model.set_api_key(task.api_key)
        self.client.api_key = task.api_key
        self.question = task.raw_task.get("question")
        self.answer = task.raw_task.get("answer")
```

#### 完整代码示例

```python
import openai
from trinity.common.workflows import WORKFLOWS, WorkflowWithRecording

@WORKFLOWS.register_module(name="example_workflow")
class ExampleWorkflow(WorkflowWithRecording):
    can_reset: bool = True

    def __init__(self, *, task: Task, model: ModelWrapper, auxiliary_models: List = None):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        self.question = task.raw_task.get("question")
        self.answer = task.raw_task.get("answer")
        self.rollout_args = task.rollout_args
        self.client = openai.AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)

    def calculate_reward(self, response: str, truth: str) -> float:
        if response == truth:
            return 1.0
        else:
            return 0.0

    async def run_async(self) -> Metrics:
        resp = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": f"Question:\n{self.question}",
                }
            ],
            temperature=self.rollout_args.temperature,
        )
        response_text = resp.choices[0].message.content
        reward: float = self.calculate_reward(response_text, self.answer)
        await self.update_reward(reward)
        return {"example/reward": reward}

    def reset(self, task: Task):
        self.task = task
        self.model.set_api_key(task.api_key)
        self.client.api_key = task.api_key
        self.question = task.raw_task.get("question")
        self.answer = task.raw_task.get("answer")
```

---

### 步骤 3：使用你的工作流

实现并注册工作流后，就可以通过将配置文件中 `buffer.explorer_input.taskset` 的 `default_workflow_type` 设置为你的工作流名称来使用它。例如：

```yaml
buffer:
  # Other fields
  explorer_input:
    taskset:
      path: /path/to/taskset
      default_workflow_type: example_workflow
      # Other fields
```

现在你可以使用以下命令在 Trinity-RFT 中运行你的工作流：

```bash
trinity run --config <your_yaml_file>
```

---

### LLM-as-a-judge 支持

LLM-as-a-judge 是一种常见的奖励计算方法，尤其适用于开放式任务（如编程、写作等）。在这类场景下，Workflow 需要借助额外的 LLM 来评估答案质量并计算奖励信号（reward）。

为此，Trinity-RFT 提供了 Auxiliary Models（辅助模型）机制。辅助模型是一组未参与训练的模型，Workflow 可利用这些模型辅助完成任务，例如作为评判者（judge）计算奖励。

你可以在配置文件中通过 `explorer.auxiliary_models` 字段指定一个或多个辅助模型。例如：

```yaml
explorer:
  auxiliary_models:
    - model_path: Qwen/Qwen2.5-32B-Instruct
      engine_num: 1
      tensor_parallel_size: 2
      enable_thinking: false
      max_prompt_tokens: 12288
      max_response_tokens: 12288
      max_model_len: 16384
    - model_path: Qwen/Qwen3-8B
      engine_num: 1
      tensor_parallel_size: 2
      enable_thinking: false
      max_prompt_tokens: 12288
      max_response_tokens: 12288
      max_model_len: 16384
```

请注意，每个辅助模型会独立占用 `tensor_parallel_size * engine_num` 个 GPU，请根据硬件资源合理配置。在启用辅助模型后，Trainer 可用的 GPU 数量为总 GPU 数量减去所有辅助模型及被训练的推理模型（`rollout_model`）所占用的 GPU 数量。

配置文件中指定的辅助模型会以 `ModelWrapper` 实例列表的形式传递给 `Workflow` 初始化方法的 `auxiliary_models` 参数。每个 `ModelWrapper` 同样暴露 `base_url`、`api_key`、`model_name`，推荐直接用它们创建 OpenAI 客户端来访问辅助模型：

```python
class MyWorkflow(WorkflowWithRecording):
    def __init__(self, *, task, model, auxiliary_models=None):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        self.judge = self.auxiliary_models[0]  # ModelWrapper
        self.judge_client = openai.AsyncOpenAI(
            base_url=self.judge.base_url, api_key=self.judge.api_key
        )

    async def run_async(self) -> Metrics:
        response = await self.do_something()
        reward_response = await self.judge_client.chat.completions.create(
            model=self.judge.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a judge. You need to give a score from 0 to 1 based on the quality of the answer.",
                },
                {
                    "role": "user",
                    "content": f"Question:\n{self.task.raw_task['question']}\nAnswer:\n{response}\nPlease give a score from 0 to 1.",
                },
            ],
            temperature=0.0,
            max_tokens=10,
        )
        # 解析奖励分数
        reward = float(reward_response.choices[0].message.content.strip())
        await self.update_reward(reward, info={"source": "llm_as_a_judge"})
        return {"my_workflow/judge_reward": reward}
```

#### 调试模式（Debug Mode）

在 Workflow 开发过程中，频繁启动完整训练流程进行测试既耗时又低效。为此，Trinity-RFT 为开发者提供了调试模式。该模式通过预先启动推理模型，能够快速运行指定的工作流并获取结果，避免因模型加载和初始化带来的重复等待，大幅提升开发效率。流程如下：

```{mermaid}
flowchart LR
    A[启动推理模型] --> B[调试 Workflow]
    B --> C[检查 Experience]
    C --> B
```

启动推理模型的命令如下：

```bash
trinity debug --config <config_file_path> --module inference_model
```

其中，`config_file_path` 为 YAML 格式的配置文件路径，格式与 `trinity run` 命令所用配置文件一致。配置文件中的 `explorer.rollout_model` 和 `explorer.auxiliary_models` 字段会被加载，用于初始化推理模型。

模型启动后会持续运行并等待调试指令，不会自动退出。此时，你可在另一个终端执行如下命令进行 Workflow 调试：

```bash
trinity debug --config <config_file_path> --module workflow --output-dir <output_dir> [--plugin-dir <plugin_dir>] [--enable-profiling] [--disable-overwrite]
```

- `<config_file_path>`：YAML 配置文件路径，通常与启动推理模型时使用的配置文件相同。
- `<output_dir>`：调试输出保存目录。如果未指定，调试输出将保存在当前工作目录下的 `debug_output` 目录中。
- `<plugin_dir>`（可选）：插件目录路径。如果你的 Workflow 或奖励函数等模块未内置于 Trinity-RFT，可通过该参数加载自定义模块。
- `--enable-profiling`（可选）：启用性能分析，使用 [viztracer](https://github.com/gaogaotiantian/viztracer) 对 Workflow 运行过程进行性能分析。
- `--disable-overwrite`（可选）：禁用输出目录覆盖功能。如果指定的文件夹非空，程序将自动创建一个带有时间戳后缀的新目录（例如 `debug_output_20251203211200`）以避免覆盖现有数据。

调试过程中，配置文件中的 `buffer.explorer_input.taskset` 字段会被加载，用于初始化 Workflow 所需的任务数据集和实例。需注意，调试模式仅会读取数据集中的第一条数据进行测试。运行上述命令后，工作流产出的 Experience 会被写入指定输出目录下的 `experiences.db` 文件中，而运行过程中记录的指标会打印在终端以便检查。

```bash
trinity debug --config <config_file_path> --module viewer --output-dir <output_dir> --port 8502
```

该命令会在 `http://localhost:8502` 启动 Experience Viewer，用于可视化调试过程中生成的 Experience。你可以在用户友好的界面中检查生成的 Experience。需注意，Viewer 会从指定输出目录下的 `experiences.db` 文件中读取 Experience，因此请确保你已成功运行过 Workflow 调试命令，且替换 `<output_dir>` 为实际的输出目录。

调试完成后，可在推理模型终端输入 `Ctrl+C` 以终止模型运行。


#### 运行时监控

在上述调试模式中，你可以快速测试和验证工作流的实现。然而，在实际训练过程中，你可能希望实时监控工作流的运行状态，以确保其按预期工作。为此，Trinity-RFT 提供了基于日志系统的监控功能。`WorkflowWithRecording` 基类内置了一个日志记录器（logger），你可以使用它来记录重要的运行时信息。

```python
class WorkflowWithRecording(WorkflowBase):

    def __init__(
        self,
        *,
        task: Task,
        model: ModelWrapper,
        auxiliary_models: Optional[List[ModelWrapper]] = None,
    ):
        # ...
        self.logger = get_logger(__name__)  # 用于运行时监控的内置 logger
```

该内置的 logger 会将日志输出到控制台和 `<checkpoint_root_dir>/<project>/<group>/<name>/log` 目录下的文件中。这样就可以方便地在训练过程中监控工作流的运行状态。由于所有 Workflow 子类均继承该 logger，因此你可以直接在自定义工作流中使用它来记录关键信息。

```python
class ExampleWorkflow(WorkflowWithRecording):
    async def run_async(self) -> Metrics:
        self.logger.info(f"Starting workflow for task: {self.task}")
        # your workflow logic
        if some_error_condition:
            self.logger.error("An error occurred during workflow execution.")
        self.logger.info(f"Completed workflow for task: {self.task}")
        return {"example/reward": reward}
```

由于 Trinity-RFT 会自动创建一组 Workflow Runners 来并行执行 Workflow。每个运行器会将其日志输出到一个单独的日志文件中。日志文件的命名规则为 `explorer_runner_<runner_id>.log`，其中 `<runner_id>` 是工作流运行器的唯一标识符。通过这种设计，你可以独立地追踪正在并行执行的每个工作流实例的运行情况。日志文件的具体组织结构如下：

```
<checkpoint_root_dir>/<project>/<group>/<name>/log/
    ├── explorer_runner_0.log
    ├── explorer_runner_1.log
    ├── explorer_runner_2.log
    └── ...
```

Trinity-RFT 还提供了一个方便的 `log` 命令来实时查看这些日志。你可以使用 `trinity log --log-dir /path/to/log/dir -k explorer_runner` 命令来过滤并查看所有 workflow runner 的日志，或者使用 `trinity log --log-dir /path/to/log/dir -k explorer_runner_0` 来查看特定 workflow runner 的日志。

---

### 附录：旧版 Workflow 接口（兼容）

对于简单的单轮任务，Trinity-RFT 仍保留旧版 `Workflow` 接口。与 `WorkflowWithRecording` 不同，旧版接口要求工作流**手动构造并返回 `Experience` 列表**，模型也不会自动 recording。所有内置工作流（`MathWorkflow` 等）目前仍基于此接口。如果你不需要复杂的 agent 循环，可以继续使用它。

旧版 `Workflow` 基类接口如下：

```python
class Workflow(WorkflowBase):

    def __init__(
        self,
        *,
        task: Task,
        model: ModelWrapper,
        auxiliary_models: Optional[List[ModelWrapper]] = None,  # 主要用于 LLM-as-a-judge 场景, 也可以用作distillation的techer
    ):
        self.task = task
        self.model = model
        self.auxiliary_model_wrappers = auxiliary_models
        self.auxiliary_models = ...  # 从 ModelWrapper 自动派生的 OpenAI client
        self.logger = get_logger(__name__)  # 用于运行时监控的内置 logger

    @abstractmethod
    def run(self) -> List[Experience]:
        """Run the workflow and return a list of Experiences."""
```

##### 初始化与 `run` 方法

`Workflow` 接受与新版相同的初始化参数（`task`、`model`、`auxiliary_models`），但 `model` 提供的是同步/异步的 `generate` 以及 `chat` 方法，返回结构包含 `response_text`、`tokens`、`prompt_length`、`logprobs`。`auxiliary_models` 则是框架自动派生的 `openai.OpenAI` / `openai.AsyncOpenAI` 客户端列表。

以下是一个手动构造 `Experience` 的简单实现：

```python
class ExampleWorkflow(Workflow):

    def __init__(self, task: Task, model: ModelWrapper, auxiliary_models: List):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        self.question = task.raw_task.get("question")
        self.answer = task.raw_task.get("answer")
        self.rollout_args = task.rollout_args

    def calculate_reward(self, response: str, truth: str) -> float:
        if response == truth:
            return 1.0
        else:
            return 0.0

    def run(self) -> List[Experience]:
        responses = self.model.chat(
            [
                {
                    "role": "user",
                    "content": f"Question:\n{self.question}",
                }
            ],
            temperature=self.rollout_args.temperature,
        )
        response = responses[0]
        reward: float = self.calculate_reward(response.response_text, self.answer)
        return [
            Experience(
                tokens=response.tokens,
                prompt_length=response.prompt_length,
                reward=reward,
                logprobs=response.logprobs,
            )
        ]
```

##### 批量重复运行

旧版 `Workflow` 支持 `can_repeat` 与 `set_repeat_times`，可在一次 `run` 内通过模型批量推理获得同一问题的多个回复（适用于 GRPO 等算法）。`set_repeat_times` 接受 `repeat_times`（执行次数）和 `run_id_base`（首次运行 ID，多轮交互场景使用）：

```python
class ExampleWorkflow(Workflow):
    can_repeat: bool = True

    def set_repeat_times(self, repeat_times, run_id_base):
        self.repeat_times = repeat_times
        self.run_id_base = run_id_base

    def run(self) -> List[Experience]:
        responses = self.model.chat(
            [
                {
                    "role": "user",
                    "content": f"Question:\n{self.question}",
                }
            ],
            n=self.repeat_times,
            temperature=self.rollout_args.temperature,
        )
        experiences = []
        for response in responses:
            reward: float = self.calculate_reward(response.response_text, self.answer)
            experiences.append(
                Experience(
                    tokens=response.tokens,
                    prompt_length=response.prompt_length,
                    reward=reward,
                    logprobs=response.logprobs,
                )
            )
        return experiences
```

##### 使用 OpenAI API 与 `extract_experience_from_history`

旧版接口下若要使用 OpenAI API 风格调用模型，可通过 `self.model.get_openai_client()`（或 `get_openai_async_client()`）获取客户端。recording 与 OpenAI API 服务由框架自动开启（无需手动配置 `enable_history` / `enable_openai_api`），框架会自动记录可训练数据，你可通过 `extract_experience_from_history` 将其提取为 `Experience` 列表：

```python
class ExampleWorkflow(Workflow):

    def __init__(self, task: Task, model: ModelWrapper, auxiliary_models: List):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        self.client: openai.OpenAI = self.model.get_openai_client()
        self.agent = MyAgent(openai_client=self.client)

    def calculate_reward(self, response: str) -> float:
        # your reward calculation logic

    def run(self) -> List[Experience]:
        response = self.agent.run()
        reward = self.calculate_reward(response)
        experiences = self.model.extract_experience_from_history()
        for exp in experiences:
            exp.reward = reward
        return experiences
```

```{tip}
1. 旧版 OpenAI API 仅自动记录 `openai.OpenAI.chat.completions.create` 及 `openai.AsyncOpenAI.chat.completions.create` 的调用历史，且不支持流式输出。
2. 调用 `chat.completions.create` 时，`model` 字段可通过 `openai_client.models.list().data[0].id` 或 `openai_client.model_path` 获取。
3. 更复杂的使用 OpenAI API 的工作流实例可参考 [ReAct Agent 训练](./example_react.md)。
```

对于旧版接口下的 LLM-as-a-judge，`auxiliary_models` 是框架自动派生的 OpenAI client 列表，可直接调用：

```python
class MyWorkflow(Workflow):
    def __init__(self, *, task, model, auxiliary_models=None):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        self.judge_model = self.auxiliary_models[0]  # 自动派生的 OpenAI client

    def run(self) -> List[Experience]:
        response = self.do_something()
        reward_response = self.judge_model.chat.completions.create(
            model=self.judge_model.model_path,
            messages=[...],
            temperature=0.0,
            max_tokens=10,
        )
        reward = float(reward_response.choices[0].message.content.strip())
        return [
            Experience(
                tokens=response.tokens,
                prompt_length=response.prompt_length,
                reward=reward,
                logprobs=response.logprobs,
            )
        ]
```
