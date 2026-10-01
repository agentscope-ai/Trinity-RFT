# 第 4 章：advantage — 为什么不用 reward 直接训

> **本章模式：拆解**。第 3 章你看到一个 task 组里有 3 条成功（reward 1.0）、13 条失败（reward -0.1）。这一章讲 GRPO 怎么把这组 reward 变成 advantage、怎么广播到每个 token，以及为什么「成功的力度比失败大 4.33 倍」反而是对的。

---

## 4.0 为什么不能直接用 reward 当训练信号

最朴素的想法：reward 高就鼓励、reward 低就抑制：

```python
loss = -reward * log_prob(trajectory)   # ❌ 朴素 REINFORCE
```

> `log_prob(动作)` = 模型输出这个动作的**概率的对数**，越大代表模型越「想」输出它。这个式子的意思是：reward 高就把对应动作的概率推大、reward 低就推小。

这在 ALFWorld 上会出问题：

| 问题 | 在 ALFWorld 的具体表现 |
|---|---|
| **量纲不可比** | `pick_and_place` 组平均 reward 0.87，`look_at_obj_in_light` 组平均 0.05。直接相乘，学习信号被简单任务支配。|
| **方差大、没 baseline** | reward 在 {-0.1, 1.0} 跳变，没减去 baseline（baseline = 一个用来判断「比平均好还是差」的参照值），学习信号噪声极大。|
| **全失败时没信号** | 一个组 16 条全失败（reward 全 -0.1），朴素 REINFORCE 会「无差别抑制所有 16 条」——但它们里有的只差一步、有的从头就错，应该区别对待。|

GRPO（Group Relative Policy Optimization）就是为解决这三个问题设计的。

---

## 4.1 一句话讲清 GRPO：用「组内平均」便宜地估算 baseline

先把一个关键概念讲清楚：**baseline（基线 / 组内平均）**。

§4.0 说「直接用 reward 当信号噪声大」。为什么？因为光有 reward，你只知道「得了几分」，不知道「这算好还是不好」。用 candle 例子：一条轨迹 reward=1.0（成功）——

- 如果这一组 16 条**全都成功**，那 1.0 只是「正常水平」，不值得特别鼓励；
- 如果 16 条里**只有这 1 条成功**、其余 15 条失败，那这条 1.0 就「远好于平均」，应该大力鼓励。

所以我们需要一个**参照值**来判断「比平均好还是差」——这就是 **baseline**。而 advantage = reward − baseline = 「比组内平均好/差多少」。

减去 baseline 是很多 RL 方法都做的，并不稀奇；真正**昂贵**的是**怎么估算这个 baseline**——之前的方法（如经典 PPO）要**额外训练一个 critic 网络**来估计它，成本高。**GRPO 的特点**正在于此：直接用**组内平均**（同一个任务 16 条 rollout 的平均 reward）当 baseline，**不用训 critic**，便宜又简单。于是：

> **GRPO 一句话**：同一个任务采 G 条轨迹（这 G 条合起来叫一个「**组 / group**」），把每条的 reward **减去组内平均（baseline）**、再除以标准差，就是 advantage。

用 candle 例子：对 `put a candle in countertop.` 采 16 条 rollout（有的成功、有的失败）——这 16 条就是一个组：

```
对 candle 任务采 G=16 条 rollout:
  rewards = [r₁, r₂, ..., r₁₆]   # 比如 3 条把 candle 放上台面(=1.0)，13 条没放上去(=-0.1)
  baseline = μ = mean(rewards)   # 组内平均 = 这组的参照值
  σ = std(rewards)
  advantageᵢ = (rᵢ - μ) / (σ + ε)   # ε=1e-6 防除零；成功 3 条→正(鼓励)，失败 13 条→负(抑制)
```

**直觉**（都在和「组内平均」比）：

- 比组内平均好 → advantage > 0 → 鼓励；
- 比组内平均差 → advantage < 0 → 抑制；
- 组内全一样（σ=0，全成功或全失败）→ advantage 全 0 → 这一组**不学**（第 3 章 §3.5 的死组）。

**对比经典 PPO**：经典 PPO 要**另外训练一个 critic 网络**来估计 baseline；GRPO 直接用「同组 16 条的平均」当 baseline，**不需要 critic**——这是 GRPO 的核心简化，也是它适合稀疏 reward 任务的原因。

---

## 4.2 GRPO 的自动平衡：稀有的成功用力奖、常见的失败轻轻罚

回到一个真实组（和 candle 例子同理：同一个任务的 16 条 rollout；`batch=163, task=13`，13 失败 + 3 成功）：

```
rewards = [-0.1 ×13, 1.0 ×3]
μ = mean = (13×(-0.1) + 3×1.0) / 16 = 0.10625
σ = std  = 0.44342   (torch.std, 无偏 n-1)
ε = 1e-6

advantage(成功 1.0) = (1.0 - 0.10625) / 0.44342 = +2.0156
advantage(失败 -0.1) = (-0.1 - 0.10625) / 0.44342 = -0.4651
```

运行 `python scripts/rl_tutorial/ch4_compute_advantage.py --sample` 可复现。读这组数：

| 原 reward | 条数 | advantage | 解释 |
|---:|---:|---:|---|
| 1.0 | 3 | **+2.016** | 强力鼓励：成功很稀有，比平均好得多 |
| -0.1 | 13 | **-0.465** | 温和抑制：失败很常见，每条只比平均差一点 |

### 🔑 本章最重要的洞察：4.33 倍 = 数量反比

注意 `|+2.016 / -0.465| = 4.33`，而 **失败/成功数量比 = 13/3 = 4.33**。**两个 4.33 完全相等**——这不是巧合：

```
总「上升推力」 = 3 条成功 × 2.016 = 6.05
总「下降推力」 = 13 条失败 × 0.465 = 6.05      ← 完全相等！
所有 advantage 之和 = 0   （因为减去了均值，这是数学必然）
```

**含义**：GRPO 自动让「稀有的成功」被**用力奖**、「常见的失败」被**轻轻罚**，但**总奖惩平衡**。这正是稀疏 reward 任务想要的：

- 成功只有 3 条，每条必须推得够狠（+2.0），模型才记得住「这么做能成」；
- 失败有 13 条，如果每条也推 -2.0，模型会被「到处是惩罚」淹没、学习信号失控；温和的 -0.465 刚好抵消。

> 💡 **这就是第 3 章末尾问题的答案**：「成功力度大 4.33 倍」恰好补偿「成功数量少 4.33 倍」，使学习信号自平衡。GRPO 不需要你调这个比例——**组内统计自动算出来了**。这也是为什么它对 reward 量纲/偏移不敏感（第 3 章 §3.0 注脚）。

---

## 4.3 multi-step GRPO：把奖惩送到每个 action token

agentic 场景的特殊点：一条轨迹有 N 个 step，每个 step 的 action 又由若干 token 组成（比如 `take candle 2 from toilet 1` 就是一串 token）。轨迹的 advantage 要**逐级送到训练真正起作用的地方**，分两步：

1. **先分到每个 step**：RL 的判断是「这**整局**好不好」（轨迹级 advantage），而一条轨迹由多步动作组成。最简单的做法是：**每一步都共享这个轨迹级 advantage**——成功轨迹的每一步都被鼓励，失败轨迹的每一步都被抑制。
2. **再送到每个 token**：训练真正调整的是**模型输出每个动作的概率**——说白了，一个动作被「鼓励」得越多，模型下次越可能输出它；被「抑制」得越多越不可能（具体怎么调整，第 5 章会讲）。而每步 action 由若干 token 组成，所以这一步的 advantage 要再送到**该动作的每个 token** 上。

两步合起来的结果：**整条轨迹的所有 action token，共享同一个 advantage**。这是**最简单、最容易想到**的一种分配方法（但不是唯一的，也不一定最好——见下文）。

```text
candle 成功轨迹（advantage = +2.016）：
  step0  action=look                  → 这些 action token 都被推高 +2.016
  step6  action=take candle 2 ...     → 推高 +2.016
  step8  action=move candle 2 ...     → 推高 +2.016
  （observation / prompt 不推——它们不是模型输出的）
```

<details>
<summary>实现细节：StepWiseGRPOAdvantageFn 源码骨架 + 三个要点（可选阅读）</summary>

```python
# StepWiseGRPOAdvantageFn.process() 的骨架
task_exps = group_by(exps, "task")                 # 1) 按 task 分组
for task_exp in task_exps.values():
    run_exps = group_by(task_exp, "run")           # 2) 组内按 run（16 条轨迹）分
    last_step_exps = {rid: steps[-1] for rid, steps in run_exps.items()}  # 3) 取每条轨迹「最后一步」的 reward
    scores, _, should_skip = self.calculate_last_step_advantage(last_step_exps)  # 4) 组内归一化算 advantage
    self.broadcast_advantages(run_exps, scores)    # 5) 把 advantage 广播回该轨迹的每一步

# broadcast_advantages 的核心一行：
exp.advantages = exp.action_mask * score           # ← 只给 action token，乘 score
```

三个实现细节（读了骨架才有意义）：

1. **用「最后一步」的 reward 代表整条轨迹**：终止 reward 已广播到每步（第 3 章 §3.2），所以最后一步的 `reward` = 轨迹终止 reward，取 `steps[-1]` 即可。
2. **`action_mask` 只选模型自己输出的 token**：observation（env 给的）和 prompt 不参与 loss——**只训练模型生成的 action token**（不想让模型去「学习预测环境的 observation」）。
3. **整条轨迹共享一个 advantage**：一条成功轨迹拿 +2.016，则它内部**所有** step 的 action token 都被推高 +2.016。

</details>

**为什么这个「粗暴」方案 work**：

| 理由 | 说明 |
|---|---|
| 回避「功劳分配」(credit assignment) | 不需要判断「candle 这一局里，到底是 `take candle` 还是 `move to countertop` 哪一步真正促成了成功」，统一推所有 action token |
| 统计有效 | 每 step 有 16 task × 16 run = 256 条轨迹同时给信号，「通常导致成功」的行为模式在统计上胜出 |
| 实测可行 | 本实验 reward -0.09 → +0.94 就是这么来的 |

> **一句话总结**：GRPO 粗暴但有效地把**成功轨迹里模型做过的每个动作都推高、失败轨迹的每个动作都压低**——统计上「通常导致成功」的动作会胜出。

> ⚠️ **但这个最简单的方法，不一定是好方法**。它的问题在于：分不清一条轨迹里**到底是哪一步真正促成了成功**——废动作（比如多余的 `look`）也跟着整条轨迹被一起推高。这个「功劳分配」(credit assignment) 问题、以及更精细的解法 **GiGPO**，放在下面的注脚里，感兴趣可点开。

> **进阶注脚：GiGPO——给每一步更细的「功劳分配」**
>
> 上面这种「整条轨迹一刀切」的做法，没法区分到底是哪一步真正促成了成功。**GiGPO** 想解决这个问题，一句话原理：**把不同轨迹里「到达过同一个环境状态」的那些步分到一组，比较在这个状态下哪个动作带来的后续回报更高，从而给这一步单独的功劳分**，而不是整条轨迹平摊。比如多条 candle 轨迹都到过「站在 toilet 1 前、还没拿 candle」这个状态，GiGPO 会比较在该状态下 `take candle` vs 瞎逛哪个后续回报高，给这一步单独的功劳分。

<details>
<summary>GiGPO 的实现细节（yaml / 指标 / 公式，可选阅读）</summary>

§4.3 的 trajectory-level 广播有个缺点：成功轨迹里的废动作也被推高。**GiGPO**（Group-in-Group PO, [arXiv:2505.10978](https://arxiv.org/abs/2505.10978)）用 `env_state_hash` 做**第二步分组**，给每个 step 更精细的 credit。

**关键**：本实验的 `StepWiseAlfworldWorkflow` **已经为 GiGPO 准备好了元数据**（第 2 章 §2.1 的 `env_state_hash` + `step_reward`），所以**换算法不用改 workflow**，只改 yaml：

```yaml
# examples/gigpo_alfworld/gigpo.yaml —— 和 baseline 只差 algorithm 段
algorithm:
  algorithm_type: gigpo          # ← 从 multi_step_grpo 换成 gigpo
  advantage_fn: gigpo
  advantage_fn_args:
    omega: 1.0                   # step-level advantage 的权重
    gamma: 1.0                   # 折扣因子
    fnorm: none                  # agent 任务默认不除 std
```

GiGPO 的 advantage 是**两层之和** `A = A_E + ω · A_S`（trinity-rft 包内 [`gigpo_advantage.py`](https://github.com/agentscope-ai/Trinity-RFT/blob/65139711219da4338c954e546c4b8434e4f75ec2/trinity/algorithm/advantage_fn/gigpo_advantage.py)）：

| 层 | 怎么分组 | 比什么 | 对应 GRPO 的什么 |
|---|---|---|---|
| **A_E**（episode 级）| 按 task，组内 16 条 run | 整条轨迹回报 R(τ)=Σr_t | = §4.2 的 GRPO advantage |
| **A_S**（step 级）| 按 `env_state_hash` **跨整个 batch** | 同一「锚点状态」下不同动作的折扣回报 R_t | GRPO 没有的额外信号 |

**A_S 的精髓**：把所有轨迹里**到达过同一个环境状态**（observation 哈希相同）的 step 分到一组，比较「在这个状态下，不同动作带来的后续回报」。比如 16 条轨迹里有 5 条都到过「站在 fridge 1 前、手里拿着 cup」这个状态——GiGPO 会比较这 5 条在该状态下的动作（`open fridge` vs `cool cup` vs 瞎搞）哪个后续回报高，**给这一步单独的 credit**，而不只是继承整条轨迹的 advantage。

**怎么判断 GiGPO 的 step-level 信号有没有生效**（gigpo.yaml 专属指标）：

```
'gigpo/anchor_group_hit_ratio': 0.x   # 有多少锚点状态被访问 ≥2 次（>0 才有 step 信号）
'gigpo/mean_abs_A_E' vs 'gigpo/mean_abs_A_S'   # episode vs step credit 的相对强度
```

> **本教程主线用 multi_step_grpo**（更简单、已验证 reward 0.94）。GiGPO 是第 6 章实验 C 的 ablation——**同一份数据、同一个 workflow，只改 algorithm 段**，对比两条曲线。这正是 Trinity「算法—数据解耦」的体现：换算法 = 改几行 yaml。
>
> 稀疏 reward 下 GiGPO 的 `anchor_group_hit_ratio` 可能不高（ALFWorld 状态空间大、哈希精确匹配，很多状态只被访问一次 → A_S=0 退化成纯 GRPO）。这是第 6 章要实测的开放问题。

</details>

---

## 4.4 advantage 健康度：怎么判断训练在正常学

两个**概念级**信号（记住结论即可，具体日志指标见折叠）：

- **死组**：组内 16 条轨迹 reward 全同（全成功或全失败）→ advantage 全 0 → 这一组**白跑**。G 越大、任务难度越分散，死组越少（第 3 章 §3.5）。
- **过更新预警**：`ppo_kl`（新旧 policy 偏移）与 `pg_clipfrac`（被 clip 的 token 占比）突然飙升 = 训练在试图大跳，是**崩溃前兆**（第 5 章 §5.5、第 6 章实验 A 用它早停）。

<details>
<summary>真实日志指标与健康范围（可选阅读）</summary>

trainer.log / explorer.log 每步打印健康度指标。下面是本实验的**真实值**：

```
# explorer.log（advantage 在 explorer 的 experience pipeline 里算，compute_in_trainer=False）
'experience_pipeline/group_advantages/reward_std/mean':  0.414   # 16 个组内 std 的均值（中段最强）
'experience_pipeline/group_advantages/reward_std/min':   0.000   # ⚠️ 总有组结果全同 → advantage 0
'experience_pipeline/skipped_group_ratio':               0.000   # 未设 std_threshold，死组不跳过只贡献 0

# trainer.log（PPO 更新健康度）
'actor/ppo_kl':       0.001 ~ 0.005   # 稳态：新旧 policy 几乎不偏移（健康）
'actor/pg_clipfrac':  0.002 ~ 0.009   # 稳态：只有 <1% 的 token 被 clip
'actor/grad_norm':    42 → 240 → 65   # 中段最大（学得最猛），后期回落
'critic/advantages/mean': ≈ 0         # advantage 均值恒 ≈0（§4.2 的「和为 0」）
```

**怎么读 + 警惕信号**：

| 指标 | 健康范围 | 异常解读 |
|---|---|---|
| `reward_std/min` | 偶尔 = 0 正常 | 大量组 = 0 → G 不够 / 任务两极分化 |
| `ppo_kl` | 0.001–0.020 | **> 0.03 → 过更新**。本实验 step 118 飙到 **0.119** = 崩溃前兆（第 5 章 §5.5）|
| `pg_clipfrac` | < 0.05 | > 0.10 → 大量 token 被 clip，lr 偏大 |
| `grad_norm` | 中段升、后期降 | 突然飙升 + reward 跌 = 训练发散 |

> 本实验稳态 `ppo_kl` 全程 0.001-0.005（极健康），但 **step 118-133 反复出现 0.03-0.29 的尖峰**，正好对应 reward 从峰值 0.936 崩回 0.31。**ppo_kl 尖峰 = 过度训练的最早预警**（第 6 章实验 A 用它做早停信号）。

</details>

---

## 4.5 动手试试

```bash
# 用真实 16-run 组算 step-wise GRPO advantage
python scripts/rl_tutorial/ch4_compute_advantage.py --sample

# 验证「平移不变」（第 3 章 §3.0 注脚）：失败 reward 从 -0.1 改成 0，advantage 不变
python scripts/rl_tutorial/ch4_compute_advantage.py --sample --shift 0.1

# 自定义：全成功组会怎样？（std=0 → advantage 全 0 → 白跑）
python scripts/rl_tutorial/ch4_compute_advantage.py --rewards 1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0

# 自定义：极端分化（1 成功 15 失败）
python scripts/rl_tutorial/ch4_compute_advantage.py --rewards 1.0,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1
```

---

## 4.6 这一章你应该带走的

- ✅ **GRPO = 减去「组内平均」(baseline)**：`advantage = (reward − 组内平均) / (组内 std + ε)`，按 task 分组、组内 16 条 run 比较；baseline 就是组内平均，**不需要 critic**（GRPO 的特点正是便宜地估算 baseline）。
- ✅ **自动平衡**：稀有的成功**用力奖**、常见的失败**轻轻罚**，力度比 = 数量反比（4.33 倍），总奖惩平衡——稀疏 reward 也能学。
- ✅ **把奖惩送到每个 action token**：两级分配（轨迹 → 每个 step → 每个 token）；这是最简单的方法，但有「分不清哪一步促成成功」(credit assignment) 的问题，GiGPO 是更细的解法（§4.3 注脚）。

❌ **你还不需要懂**：

- advantage 怎么进入训练：PPO clip、KL、entropy、loss、梯度、权重更新（→ 第 5 章）

💡 **留给自己的问题**：advantage 算好了。但它怎么真正进入训练——为什么需要 clip、loss 怎么算、梯度怎么改变权重？第 5 章把这最后一段拆开：advantage → loss → 梯度 → 新权重。

---

**上一章**：[第 3 章：reward 怎么算](./ch3_reward怎么算.md) ｜ **下一章**：[第 5 章：权重怎么更新](./ch5_loss与权重更新.md)
