# RL Tutorial：跑通并看懂强化学习（以 ALFWorld 为实验环境）

> 📌 这是一份**面向零 RL 基础读者**的强化学习入门教程。主线不是背公式，而是：**先亲手跑一个真实训练、亲眼看到 RL 让模型变强**，再一层层把「rollout → reward → advantage → loss → 权重更新」拆开讲清楚。实验环境用 [ALFWorld](https://github.com/alfworld/alfworld)（一个文字版家务具身环境），它只是我们的**数据集/实验台**——所有 RL 概念都在这个可跑、可看、可改的真实训练里讲。

> 7 章按「黑盒 → 拆解 → 实验」的顺序。前五章的曲线、轨迹和成功率来自同一次**实际训练**（133 step、8×A100、reward 从 -0.09 涨到 +0.94 又因过度训练崩回 +0.31），配套脚本可以用预置样本离线查看计算过程。第六章推荐你运行 **LoRA32 + TuFT 实践**，提供一键运行示例、参考曲线和留出任务评测结果；后半章提供开放实验：给出问题和配置思路，由你探索与验证，不提供参考结果。

> 🏠 **教程仓库**：[Trinity-RFT 的 `tutorial` 分支](https://github.com/agentscope-ai/Trinity-RFT/tree/tutorial)（网页版 [agentscope-ai.github.io/agentic-rl](https://agentscope-ai.github.io/agentic-rl)）；底下用到的 RL 框架是 [Trinity-RFT](https://github.com/agentscope-ai/Trinity-RFT)，作为依赖包引入（见 `pyproject.toml`）。

> **配套代码**：全参数与 LoRA32 使用同一个 [Trinity 版本](https://github.com/agentscope-ai/Trinity-RFT/commit/65139711219da4338c954e546c4b8434e4f75ec2)，各自的运行脚本负责准备环境和配置。

> 💡 **离线 HTML 版**：本目录内置构建好的网页版教程（入口 [`index.html`](./index.html)，带章节侧边导航 + 页内目录 + 代码高亮，完全离线可用）。markdown 与网页版一一对应；md 更新后运行 [`build_html.py`](./build_html.py) 可重新生成。

---

## 整体结构：跟着 RL 的概念走

7 章按「黑盒 → 拆解 → 实验」递进：章节链接与学习路径合并在同一张表里，每一章都回到同一条真实训练曲线和同几条真实轨迹，概念不悬空。

| 章 | 标题 | 这一章解决什么概念 | 模式 | 大约耗时 |
|---|---|---|---|---|
| [第 0 章](./ch0_什么是强化学习.md) | 什么是强化学习？你的任务适合吗？ | 建立 vocabulary：state / action / reward / policy | 概念 / 决策 | 10 分钟阅读 |
| [第 1 章](./ch1_跑通.md) | 跑通：先看 RL 真的有用 | 黑盒模式，看 reward -0.09 → +0.94，建立直觉 | 黑盒 / 操作 | 约 1 小时装环境 + 约 15 小时训练（历史参考） |
| [第 2 章](./ch2_rollout与experience.md) | rollout：state / action / reward 与 experience | 一次交互循环里发生了什么 | 拆解 | 15 分钟阅读 |
| [第 3 章](./ch3_reward怎么算.md) | reward：环境怎么给分 | RL 唯一的监督信号 | 拆解 | 15 分钟阅读 |
| [第 4 章](./ch4_advantage怎么来.md) | advantage：为什么不用 reward 直接训 | baseline / 组内相对 / GRPO | 拆解 | 20 分钟阅读 |
| [第 5 章](./ch5_loss与权重更新.md) | loss 与更新：advantage 怎么变成梯度 | policy gradient / PPO clip / KL / entropy | 拆解 | 20 分钟阅读 |
| [第 6 章](./ch6_动手实验.md) | LoRA 实测与开放实验 | LoRA32 推荐实践与参考结果；开放实验 | 实验 | 推荐动手运行，也可先阅读参考结果 |

---

## 写作约定

每一章末尾都有：

- ✅ **这一章你应该带走的**（3–5 条）
- ❌ **你还不需要懂**（指明哪些内容延后到哪一章）
- 💡 **留给自己的问题**（自然过渡到下一章）

**这样你永远知道**：当前该理解到什么程度（不焦虑）、没懂的部分会在哪一章解决（不挫败）、下一章为什么要读（不迷茫）。

> 我们刻意**不深入框架/工程实现**（分布式分片、显存优化、版本补丁等）。那些是「跑更大实验」时才需要的工程知识，会淹没 RL 的主线；个别确有必要提到的（如显存不够时的 offload），只做成鼠标悬停提示。

---

## 一句话讲清这次实验

> 把 **Qwen3-1.7B** 放进 **ALFWorld** 文字具身环境，让它自己摸索「找到物体 → 加热/冷却/清洗 → 放到目标位置」这类多步家务任务；用 **GRPO**（一种强化学习算法）在 **8×A100** 上训练，每个任务采 **16 条**轨迹做组内相对比较。训练平均 reward 从 **−0.09 涨到 +0.94**——但**训练过头后崩回 +0.31**。这里的平均按训练样本统计，不能直接当作每局等权的成功率；第 3 章解释两者的区别。这条「涨上去又掉下来」的完整曲线，就是本教程最好的教材：它既证明 RL 有用，也教会你 RL 的边界。

---


## 小机器人任务动画

[打开二维任务播放器](./alfworld_robot.html)，认识 ALFWorld 环境并看模型如何一步步完成任务（第 0 章 §0.2 和第 2 章 §2.1–2.3 已内嵌同一个播放器）。

<details>
<summary>播放器使用说明（可选阅读）</summary>

播放器支持暂停、逐步查看、拖动进度、调整速度，以及切换成功和失败样本。离开视野或切到后台时会暂停；网页也可离线使用。

- 小机器人代表模型；模型实际输入是文字观察。二维布局是教学示意，物品随真实观察逐步出现。
- 成功与失败样本的指令相同，房间不同。不能当作相同初始状态的严格前后对照。
- 动作和观察来自真实轨迹；最终一步没有保存执行后的观察，成功放置按终止奖励示意。

</details>

---

## 反馈

发现教程里的错误 / 不清楚的地方，欢迎在 [Trinity-RFT issue 页面](https://github.com/agentscope-ai/Trinity-RFT/issues)反馈（请注明 RL Tutorial），或向 `tutorial` 分支提交 PR。让强化学习从「少数人的黑魔法」变成「每个工程师都能上手并看懂的方法」，需要每个读者的反馈。

---

## 附录：配套素材

<details>
<summary>教程用到的图片、脚本与配置文件清单（可选阅读）</summary>

| 文件 | 用途 | 对应章 |
|---|---|---|
| [`index.html`](./index.html) / `chX_*.html` | 离线网页版教程（入口 index.html）| 全部 |
| [`build_html.py`](./build_html.py) | 把 markdown 重新构建成离线 HTML | 全部 |
| [`ch1_reward_curve.png`](./ch1_reward_curve.png) | 真实 reward 曲线（含 entropy / ppo_kl 失稳面板）| ch1 / ch5 |
| [`plot_reward_curve.py`](../../scripts/rl_tutorial/plot_reward_curve.py) | 从 `trainer.log` 重画上图 | ch1 |
| [`ch2_inspect_trajectory.py`](../../scripts/rl_tutorial/ch2_inspect_trajectory.py) | 打印一条真实轨迹的 obs→action 序列，可对比失败/成功 | ch2 |
| [`ch3_reward_to_success.py`](../../scripts/rl_tutorial/ch3_reward_to_success.py) | reward ↔ 成功占比换算（区分 turn 与 episode）+ reward 分布 | ch3 |
| [`ch4_compute_advantage.py`](../../scripts/rl_tutorial/ch4_compute_advantage.py) | 用真实 16-run 组算 GRPO advantage | ch4 |
| [`ch6_task_family_breakdown.py`](../../scripts/rl_tutorial/ch6_task_family_breakdown.py) | 按 6 个任务族分解成功率（早 vs 晚）| ch6 |
| [`examples/grpo_alfworld_general_multi_step/alfworld.yaml`](../../examples/grpo_alfworld_general_multi_step/alfworld.yaml) | 本实验的训练配置（baseline）| ch1 / ch6 |

</details>

---

**开始阅读**：[第 0 章：什么是强化学习？](./ch0_什么是强化学习.md)
