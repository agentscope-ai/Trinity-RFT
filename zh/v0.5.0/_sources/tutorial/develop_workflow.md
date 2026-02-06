(Workflows)=
## Workflow 开发指南

在 Trinity-RFT 中，工作流（Workflow）是定义 Agent 与 Environment 之间交互的核心组件。
一个合格的工作流需要使用被训练模型完成指定任务，并从环境中获取反馈信息（奖励）。本节将会介绍如何开发一个新的工作流。

---

### 步骤 0：基本概念

在开发之前，理解以下几个核心概念非常重要：

```{mermaid}
flowchart LR
    A([Task]) & B([Model]) --> C[Workflow]
    C --> D([Experience])
```

- **任务（Task）** ({class}`trinity.common.workflows.Task`)：结构化的数据实例，包含了工作流一次运行所需的各种信息。一般情况下由训练数据集提供，数据集中的每个样本都会被转化为一个 `Task` 实例。`Task` 的内容根据任务类型而异：
  - **数学问题**：包含问题和答案。
  - **编程场景**：包含题目的描述、测试用例、运行环境等复杂信息。

- **模型（Model）** ({class}`trinity.common.models.model.ModelWrapper`)：被训练的模型，工作流内需要使用该模型来执行推理。该实例由 Trinity-RFT 自动提供，支持同步以及异步的 `generate` 以及 `chat` 等方法，同时也提供了 OpenAI API 接口，能够兼容大部分 Agent 框架。

- **工作流（Workflow）** ({class}`trinity.common.workflows.Workflow`)：定义了 Agent 与 Environment 的交互流程。`Workflow` 通过 `Task` 中提供的信息初始化自身，并借助 `Model` 来执行其中定义好的交互流程。与常规 Agent 应用不同的是，工作流内部还需要计算奖励信号（reward）以指导训练过程。Trinity-RFT 包含多个内置工作流：
  - `MathWorkflow` ({class}`trinity.common.workflows.MathWorkflow`)：用于数学场景，将问题提交给 LLM，解析 LLM 响应，并计算分数（奖励）。
  - `WebShopWorkflow` ({class}`trinity.common.workflows.WebShopWorkflow`)：用于 webshop 场景，包含与环境的多轮交互。
  - `AgentScopeReActWorkflow` ({class}`trinity.common.workflows.AgentScopeReActWorkflow`)：直接使用现有的 ReActAgent（基于 AgentScope）来解决问题。

- **经验（Experience）** ({class}`trinity.common.experience.Experience`)：`Workflow` 的运行产出。产出的数量以及内部数据格式取决于所使用的训练算法。例如，对于常见的 PPO/GRPO 算法，`Experience` 包含 token ID 列表、动作掩码（标识哪些 token 是由 LLM 生成的）、每个 token 的对数概率（logprobs）、奖励信号（reward）等。

---

### 步骤 1：准备任务数据集

任务数据集通过 YAML 配置文件中的 `buffer.explorer_input.taskset` 配置项加载。
为处理 `Task` 内容的差异，Trinity-RFT 提供了一个统一的 `Task` 接口，包含以下字段：

- **`workflow`** (`str`)：你的工作流类的注册名称。你可以在 YAML 配置文件的 `buffer.explorer_input.taskset.default_workflow_type` 中指定。
- **`reward_fn`** (`Optional[str]`)：你的奖励函数的注册名称。你可以在 `buffer.explorer_input.taskset.default_reward_fn_type` 中指定。注意某些工作流已内置奖励计算；此时可省略该字段。
- **`raw_task`** (`Dict`)：原始数据的记录，以 `Dict` 格式存储。对于高度定制化的工作流，你可以直接使用 `raw_task` 初始化 `Workflow` 实例，而不依赖以下字段。
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
      default_workflow: "math_workflow"
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

`Workflow` 基类接口如下：

```python
class Workflow(ABC):

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

    @abstractmethod
    def run(self) -> List[Experience]:
        """Run the workflow and return a list of Experiences."""
```

#### 初始化你的工作流

`Workflow` 接受以下初始化参数：

- `task`({class}`trinity.common.workflows.Task`)：数据集中的单个任务。
- `model`({class}`trinity.common.models.model.ModelWrapper`)：正在训练的模型，提供类似于 OpenAI 的接口，能够接收对话消息列表并返回 LLM 生成的内容（包括回复文本 `response_text`、完整序列 token id `tokens`、prompt 部分 token 长度 `prompt_length`，以及输出 token 对数概率列表 `logprobs`）。
- `auxiliary_models`(`List[ModelWrapper]`)：辅助模型的 ModelWrapper 列表。可通过 `self.auxiliary_models` 访问 OpenAI client（根据 workflow 的 `is_async` 自动派生）。

以下是一个仅使用 `raw_task` 和 `rollout_args` 初始化简单工作流的示例。在更复杂的情况下，你可以使用 `format_args` 进行进一步自定义。

```python
class ExampleWorkflow(Workflow):

    def __init__(self, task: Task, model: ModelWrapper, auxiliary_models: List):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        self.question = task.raw_task.get("question")
        self.answer = task.raw_task.get("answer")
        self.rollout_args = task.rollout_args
        # Optional: If you want to use OpenAI API in your workflow
        # self.openai_client = self.model.get_openai_client()
```

#### 实现 `run` 方法

`run` 方法是工作流的核心方法。该方法没有输入参数，返回一个 `Experience` 列表。
以下是一个数学工作流的简单实现。

我们首先调用模型，使用给定的问题和 rollout 参数生成答案。
然后使用 `calculate_reward` 函数计算答案的奖励。
最后，我们将生成的答案和奖励封装为`Experience` 实例并返回。

```python
class ExampleWorkflow(Workflow):

    # the __init__ function

    def calculate_reward(self, response: str, truth: str) -> float:
        if response == truth:
            return 1.0
        else:
            return 0.0

    def run(self) -> List[Experience]:
        # call the model to generate multiple responses
        responses = self.model.chat(
            [
                {
                    "role": "user",
                    "content": f"Question:\n{self.question}",
                }
            ],
            temperature=self.rollout_args.temperature,
        )
        response = responses[0]  # there is only one response
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

#### 注册你的工作流

为了让 Trinity-RFT 能够通过配置文件中的名称自动找到你的工作流，你需要在 `trinity/common/workflows/__init__.py` 中的 `default_mapping` 中注册。

```python
WORKFLOWS = Registry(
    "workflows",
    default_mapping={
        "example_workflow": "trinity.common.workflows.workflow.ExampleWorkflow",
    },
)
```

#### 性能调优

以下是一些可选的性能调优方法，能够提升工作流的运行效率。当然，这些方法并非所有工作流都需要实现，具体取决于你的工作流设计。

##### 避免重复初始化

对于较为复杂的工作流，每次重新初始化会带来额外计算开销。
此时，你可以设置 `can_reset` 属性并实现 `reset` 方法以避免重复初始化。

`can_reset` 是一个类属性，表示工作流是否支持轻量化重置。

`reset` 方法接受一个新的 `Task` 实例，并使用该实例更新工作流的状态。

```python
class ExampleWorkflow(Workflow):
    can_reset: bool = True

    # some code
    # ...

    def reset(self, task: Task):
        self.question = task.raw_task.get("question")
        self.answer = task.raw_task.get("answer")
```

##### 批量运行推理任务

当前流行的很多 RL 算法需要多次运行同一个任务(例如 GRPO)。该场景下一些简单任务可以直接通过模型批量推理来获得一个问题的多个回复以提升效率。
针对该情况，你可以设置 `can_repeat` 属性并实现 `set_repeat_times` 方法。

`can_repeat` 是一个类属性，指示工作流是否支持在 `run` 方法内多次执行。

`set_repeat_times` 方法接受两个参数：`repeat_times` 指定了在 `run` 方法内需要执行的次数，`run_id_base` 是一个整数，用于标识多次运行中第一次的运行 ID，之后各次的 ID 基于此递增（该参数用于多轮交互场景，单次模型调用即可完成的任务可以忽略该项）。

```python
class ExampleWorkflow(Workflow):
    can_repeat: bool = True
    # some code

    def set_repeat_times(self, repeat_times, run_id_base):
        self.repeat_times = repeat_times
        self.run_id_base = run_id_base

    def run(self) -> List[Experience]:
        # call the model to generate multiple responses
        responses = self.model.chat(
            [
                {
                    "role": "user",
                    "content": f"Question:\n{self.question}",
                }
            ],
            n=self.repeat_times,  # run multiple times in one call
            temperature=self.rollout_args.temperature,
        )
        experiences = []
        for response in responses:
            # calculate reward
            reward: float = self.calculate_reward(response.response_text, self.answer)
            # construct Experience
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


#### 完整代码示例

```python
class ExampleWorkflow(Workflow):
    can_reset: bool = True
    can_repeat: bool = True

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
        # call the model to generate multiple responses
        responses = self.model.chat(
            [
                {
                    "role": "user",
                    "content": f"Question:\n{self.question}",
                }
            ],
            n=self.rollout_args.n,
            temperature=self.rollout_args.temperature,
        )
        experiences = []
        for response in responses:
            # calulcate reward
            reward: float = self.calculate_reward(response.response_text, self.answer)
            # construct Experience
            experiences.append(
                Experience(
                    tokens=response.tokens,
                    prompt_length=response.prompt_length,
                    reward=reward,
                    logprobs=response.logprobs,
                )
            )
        return experiences

    def reset(self, task: Task):
        self.question = task.raw_task.get("question")
        self.answer = task.raw_task.get("answer")

    def set_repeat_times(self, repeat_times, run_id_base):
        self.repeat_times = repeat_times
        self.run_id_base = run_id_base
```

---

### 步骤 3：使用你的工作流

实现并注册工作流后，就可以通过将配置文件中 `buffer.explorer_input.taskset` 的 `default_workflow_type` 域设置为你的工作流名称来使用它。例如：

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

### 其他进阶特性

#### async 支持

本节样例主要针对同步模式，如果你的工作流需要使用异步方法（例如异步 API）,你可以将 `is_async` 属性设置为 `True`，然后实现 `run_async` 方法，在这种情况下不再需要实现 `run` 方法，并且初始化参数 `auxiliary_models` 也会自动变为 `List[openai.AsyncOpenAI]` 类型，其余方法和属性保持不变。

```python
class ExampleWorkflowAsync(Workflow):

    is_async: bool = True

    async def run_async(self) -> List[Experience]:
        # your async code here

    # no need to implement run() method
```

#### 使用 OpenAI API

Trinity-RFT 的 Model 提供了 OpenAI API 接口，能够降低模型推理部分的学习成本并简化工作流的实现。

为了激活 OpenAI API 服务，你需要将配置文件中 `explorer.rollout_model.enable_openai_api` 设置为 `true` 。这样就可以通过 `Model` 实例的 `get_openai_client` 方法获取 `openai.OpenAI` 实例。

另外，由于 OpenAI API 无法提供训练所需的各项数据，你还需要将 `explorer.rollout_model.enable_history` 设置为 `true`，让框架自动记录可用于训练的数据并转化为 `Experience` 列表。你可以通过 `extract_experience_from_history` 方法来提取这些可用于训练的数据。


```yaml
# example config snippet
explorer:
  rollout_model:
    enable_openai_api: true
    enable_history: true
    # Other fields
```

```python
class ExampleWorkflow(Workflow):

    def __init__(self, task: Task, model: ModelWrapper, auxiliary_models: List):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        self.model = model
        self.client: openai.OpenAI = self.model.get_openai_client()
        # or async client
        # self.client: openai.AsyncOpenAI = self.model.get_openai_async_client()
        self.agent = MyAgent(openai_client=self.client)

    def calculate_reward(self, response: str) -> float:
        # your reward calculation logic

    def run(self) -> List[Experience]:
        # run your agent
        response = self.agent.run()
        # calculate reward
        reward = self.calculate_reward(response)
        # extract experiences from history recorded in self.model
        experiences = self.model.extract_experience_from_history()
        for exp in experiences:
            exp.reward = reward
        return experiences
```

```{tip}
1. 当前的 OpenAI API 仅会自动记录 `openai.OpenAI.chat.completions.create` 以及 `openai.AsyncOpenAI.chat.completions.create` 方法的调用历史并转化为 `Experience` 结构，且不支持流式输出。
2. 调用 `chat.completions.create` 时，其中的 `model` 字段可通过 `openai_client.models.list().data[0].id` 或 `openai_client.model_path` 获取。
3. 更复杂的使用 OpenAI API 的工作流实例可参考 [ReAct Agent 训练](./example_react.md)。
```

#### LLM-as-a-judge 支持

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

配置文件中指定的辅助模型会自动激活 OpenAI API，并将对应的 `openai.OpenAI` 或 `openai.AsyncOpenAI` 实例 (取决于 `is_async`) 传递给 `Workflow` 初始化方法的 `auxiliary_models` 参数。例如：

```python
class MyWorkflow(Workflow):
    def __init__(
        self,
        *,
        task: Task,
        model: ModelWrapper,
        auxiliary_models: Optional[List[ModelWrapper]] = None,
    ):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        self.judge_model = self.auxiliary_models[0]  # 从 ModelWrapper 自动派生的 OpenAI client

    def run(self) -> List[Experience]:
        response = self.do_something()
        reward_response = self.judge_model.chat.completions.create(
            model=self.judge_model.model_path,
            messages=[
                {
                    "role": "system",
                    "content": "You are a judge. You need to give a score from 0 to 1 based on the quality of the answer.",
                },
                {
                    "role": "user",
                    "content": f"Question:\n{self.task.raw_task['question']}\nAnswer:\n{response.response_text}\nPlease give a score from 0 to 1.",
                },
            ],
            temperature=0.0,
            max_tokens=10,
        )
        # 解析奖励分数
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

调试过程中，配置文件中的 `buffer.explorer_input.taskset` 字段会被加载，用于初始化 Workflow 所需的任务数据集和实例。需注意，调试模式仅会读取数据集中的第一条数据进行测试。运行上述命令后，工作流的返回 Experience 会被写入指定输出目录下的 `experiences.db` 文件中，而运行过程中记录的指标会打印在终端以便检查。

```bash
trinity debug --config <config_file_path> --module viewer --output-dir <output_dir> --port 8502
```

该命令会在 `http://localhost:8502` 启动 Experience Viewer，用于可视化调试过程中生成的 Experience。你可以在用户友好的界面中检查生成的 Experience。需注意，Viewer 会从指定输出目录下的 `experiences.db` 文件中读取 Experience，因此请确保你已成功运行过 Workflow 调试命令，且替换 `<output_dir>` 为实际的输出目录。

调试完成后，可在推理模型终端输入 `Ctrl+C` 以终止模型运行。
