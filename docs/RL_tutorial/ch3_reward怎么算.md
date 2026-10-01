# 第 3 章：reward — 环境怎么给分

> **本章模式：拆解**。第 2 章你看到一条轨迹末尾会拿 `1.0`（成功）或 `-0.1`（失败），而且这个值被广播到轨迹的每一步。这一章讲：reward 从哪来、为什么这么稀疏还能学、以及一个**反直觉的事实**——那个 `-0.1` 的负号其实不改变 advantage。

---

## 3.0 reward 是 RL 唯一的监督信号

回顾第 0 章：RL 没有「标准答案」，只有「分数」。这个分数就是 **reward**——它告诉模型「刚才那一整局做得好不好」，是训练**唯一**的监督信号。reward 给得好不好，直接决定 RL 能不能学、学得多快。

reward 有两种典型形态——用 candle 例子看区别：

- **稀疏 reward（sparse）**：只在**整局结束**给一个分。candle 例子里，只有「candle 放上台面」这一下算数（1.0），中间的 `take candle`、`go to countertop` 等步骤**全是 0 分**。
- **密集 reward（dense）**：**每完成一个子目标就给一点分**（假想对照）。candle 例子里可以是：找到 candle +0.3、拿起 candle +0.3、放上台面 +0.4——分段给分，「半成品」也有学习信号。

| 形态 | 在 candle 例子里 | 信号强度 |
|---|---|---|
| **稀疏（sparse）** | 只在 candle 放上台面时给分（1.0 / -0.1），中间步全 0 | 弱：要靠 16 条 rollout 的差异挤信号 |
| **密集（dense）** | （假想）找到 candle +0.3、拿起 +0.3、放上 +0.4 | 强：半成品也有学习信号 |

本实验的 ALFWorld 是**稀疏终止分**：判分由环境内置，任务完成 =1.0、没完成 =-0.1，**中间步 reward 全是 0**，你一行判分器都不用写。

> 📎 **顺带回答「为什么失败是 -0.1 而不是 0」**（第 2 章末尾的问题）：这个负号**不影响训练**——RL 用的是组内相对比较（advantage，第 4 章），给组内所有 reward 加同一个常数不改变比较结果，所以 -0.1 还是 0 都一样。-0.1 只是**让曲线更可读**（全失败时显示负值，比 0 醒目）。第 4 章讲 advantage 时展开。

这一章的核心问题就是：**只有 {-0.1, 1.0} 两档的稀疏 reward，为什么还能让训练分数逐步提高？**（答案在 §3.4 的 G=16 组内差异，以及第 4 章的 advantage。）

---

## 3.1 第一层：env 怎么给 reward

ALFWorld 的 reward 由**环境**判定，逻辑很简单（伪代码）：

```text
final_reward = -0.1                 # 默认：没完成 = 失败
for step in 1 .. 30:
    action = policy(state)
    state, reward, done = env.step(action)
    if done:                        # 任务完成！
        # candle 例子：move candle 2 to countertop 1 后，candle 放上台面 → done=True
        final_reward = 1.0          # 成功
        break
# 轨迹结束：final_reward 就是这局的 reward
    # candle 例子：放上台面 → 1.0；§2.2 那条 30 步没放上去 → -0.1
```

**什么叫「任务完成」（`done=True`）**？用 §2.2 / §2.3 贯穿始终的例子：任务是 `put a candle in countertop.`——**当 candle 被放到 countertop 上时，环境就判定任务完成**（`done=True`），给 `reward=1.0`。§2.3 里 step 8 `move candle 2 to countertop 1` 之后任务就完成了。反之，30 步内 candle 一直没放到 countertop（比如 §2.2 拿错了 toiletpaper、卡在死循环），就是没完成，`reward=-0.1`。

> 你不需要写任何判分器——环境内置了「目标达成」的判定，这就是 §0.3 说的「可验证 reward」。ALFWorld 是**二值成功判定**：要么完成（1.0），要么没完成（-0.1），没有「完成一半给 0.5 分」——这就是稀疏 reward 的根源。

<details>
<summary>reward 产生与广播的源码（可选阅读）</summary>

```python
def run(self):
    self.final_reward = -0.1          # ← 初始化为 -0.1（默认失败）
    ...

def step(self, step_num):
    ...
    observation, reward, done, info = self.env.step(action)   # env 返回即时 reward
    self._step_meta.append((env_state_hash, float(reward)))    # 存「即时 reward」= step_reward
    self.done = done
    if self.done:
        self.final_reward = reward     # ← 任务完成时，env 给 reward=1.0
    return not self.done

def reward(self, exps):
    return self.final_reward           # ← 整条轨迹的终止 reward
```

</details>

---

## 3.2 第二层：终止 reward 广播到每一步

第 2 章 §2.5 讲过：一条轨迹的**终止 reward 会广播（复制）到它的每一个 experience 上**——这样每个训练样本都携带「这局最终成没成」。（具体实现见下方折叠。）

<details>
<summary>reward 广播的源码（可选阅读）</summary>

```python
reward = self.reward(experiences)      # = final_reward = 1.0 或 -0.1
for exp in experiences:
    exp.reward = reward                # ← 轨迹里每一步的 exp.reward 都 = 这个终止值
    exp.metrics["actual_env_steps"] = step + 1
```

</details>

所以一条成功轨迹的 30 个（或 10 个）experience，**每个的 `reward` 字段都是 1.0**；失败轨迹每个都是 -0.1。

**为什么这么设计**：multi-step GRPO 要在「轨迹级别」做组内比较（第 4 章），它需要每条轨迹有一个统一的标量 reward。把终止 reward 广播到每步，是最简单的「让每个 experience 都携带轨迹结果」的方式。

---

## 3.3 reward 能换算什么：先分清「每局」和「每步」

reward 只有 {-0.1, 1.0} 两档，所以平均值可以换算成成功占比。但**先要问：这个平均值是按什么单位算的？**

- **每局（episode）等权**：每完成一局只记一次终局 reward。换算得到的就是任务局成功率。
- **每条训练样本（turn / experience）等权**：终局 reward 已广播到这一局的每个 turn；局越长，贡献的样本越多。换算得到的是训练 batch 中来自成功轨迹的 turn 占比。

本教程的 `critic/score/mean` 属于第二种。它先汇总每条 experience 上的 reward，再对 batch 中的 experience 求平均，**不是每局等权成功率，也不是 held-out 评测精度**。

**一个例子**：一局成功，用了 10 个 turn；一局失败，用了 30 个 turn。

| 统计单位 | 平均 reward | 换算后的成功占比 |
| --- | --- | --- |
| 每局等权，共 2 局 | `(1.0 − 0.1) / 2 = 0.45` | `1 / 2 = 50%` |
| 每个 turn 等权，共 40 个 turn | `(10 × 1.0 − 30 × 0.1) / 40 = 0.175` | `10 / 40 = 25%` |

公式相同，**分母不同，含义就不同**。令 `p` 为当前统计单位下的成功占比：

```text
mean_reward = p × 1.0 + (1 − p) × (−0.1) = 1.1p − 0.1
p = (mean_reward + 0.1) / 1.1
```

用这个公式换算第 1 章的曲线，得到的是 **turn 加权占比**（运行 `python scripts/rl_tutorial/ch3_reward_to_success.py --sample` 可查看）：

| trainer step | `critic/score/mean` | 来自成功轨迹的 turn 占比 |
|---:|---:|---:|
| 1 | -0.094 | **0.5%** |
| 20 | +0.012 | 10.2% |
| 40 | +0.252 | 32.0% |
| 60 | +0.524 | 56.7% |
| 80 | +0.905 | 91.4% |
| 100 | +0.768 | 78.9% |
| **115（峰值）** | **+0.936** | **94.2%** |
| 133（崩溃末）| +0.308 | 37.1% |

任务族样本中的每局成功率另有统计来源，不能用它与这张换算表“数值接近”来相互验证。比较任务能力时，应另外查看每局等权的评测结果，并说明任务集和评测条件；训练曲线帮助我们观察训练过程。

---

## 3.4 核心问题：稀疏 reward 为什么还能学

回到 §3.0 的问题。直觉上，只有 {-0.1, 1.0} 两档、中间步全 0，信号应该很弱。为什么能学起来？**靠 G=16 的组内差异**。

看一个真实的 task 组（同一个 candle 类任务的 16 条 rollout 的终止 reward）：

```
rewards = [-0.1 ×13, 1.0 ×3]   # 3 条成功（candle 放上台面），13 条失败
```

**关键**：虽然每条轨迹的 reward 只有两档，但**一个组里 16 条轨迹的成功/失败是混合的**（3 成功 + 13 失败）。RL 就能利用这个对比：**成功的 3 条被鼓励、失败的 13 条被抑制**。这个「鼓励/抑制」的信号在第 4 章叫 **advantage**，到那里再细讲——现在你只需要懂一件事：**只要组内既有成功又有失败，RL 就有东西可学**。

**对比：如果 G=1 会怎样**？单条轨迹没有「组」可比，RL 分不清「哪条更好」——**完全学不动**。这就是为什么 ALFWorld 必须用大 G：

| G（repeat_times）| 一组里出现「成功+失败混合」的概率 | 能否学 |
|---:|---|---|
| 1 | 0%（永远只有一条）| ❌ 没有差异，学不动 |
| 4 | 低（base 成功率 10% 时，4 条全失败概率 = 0.9⁴ = 66%）| ⚠️ 一半以上的组没有差异 |
| **16** | 高（16 条全失败概率 = 0.9¹⁶ = 18.5%）| ✅ 81% 的组有差异 |

→ **G=16 是稀疏 reward 的解药**：它保证大部分 task 组里既有成功也有失败，RL 才有对比可学。代价是每 step 要采 256 条轨迹（wall time 变长）。

## 3.5 稀疏 reward 的代价：「16 条结果全一样」的死组

§3.4 说：组内既有成功又有失败，RL 才有东西可学。反过来——**如果一个组的 16 条轨迹结果全一样**（全成功，或全失败），组内就没有「差异」，RL 分不清好坏，**这个组这一步就白跑了**。这种组叫「**死组**」。

死组有两种来源：

- **全失败**（训练早期常见）：难任务 base 一条都做不对（比如 `look_at_obj_in_light` 初始成功率才 2.6%），16 条全失败。这正是第 0 章 §0.3 强调「初始成功率必须 > 0」的原因——如果**所有**组都全失败，整个 batch 都没有差异，训练完全不动。本实验幸运的是简单族（`pick_and_place` 27.8%）从一开始就有正样本，先把模型「点亮」，再逐渐攻克难族。
- **全成功**（训练后期常见）：大部分任务都学会了，16 条全成功（`pick_and_place` 后期 97% 就是这样），也是死组。所以**后期死组越来越多、可学的差异越来越少**——这是「后期该停」的原因之一（第 6 章实验 A）。

> G 越大、任务难度越分散，死组越少。G=16 已把「全失败死组」的概率压到 18.5%（§3.4）。

<details>
<summary>对应的训练日志指标（reward_std / skipped_group_ratio，可选阅读）</summary>

死组在训练日志里体现为「组内 reward 的标准差 = 0」。explorer.log 每个 explore step 会打印：

```
'experience_pipeline/group_advantages/reward_std/min':   0.000   # ⚠️ 有组 std=0 → 16 条结果全同（死组）
'experience_pipeline/group_advantages/reward_std/mean':  0.414   # 组内 std 均值（健康）
'experience_pipeline/group_advantages/reward_mean/min': -0.100   # 最难的组全失败 → mean=-0.1
'experience_pipeline/skipped_group_ratio':               0.000   # 本实验没设 std_threshold，死组不跳过、只贡献 0
```

| 指标 | 健康 | 异常信号 |
|---|---|---|
| `reward_std/mean` | > 0.2 | < 0.05 → 大部分组趋同，信号枯竭（后期会这样，该停了）|
| `reward_std/min` | 偶尔 = 0 正常 | 大量组 = 0 → G 不够大，或任务难度两极分化 |

> 本实验 `reward_std/mean` 从 step1 的 0.034 升到中段 ~0.41（信号最强），再回落到后期 ~0.06-0.15（大部分任务都会做了，组内趋同）——这条曲线本身就在提示「什么时候该停」。第 6 章实验 C 会试「设 std_threshold 主动跳过死组」。

</details>

---

## 3.6 动手试试

```bash
# reward ↔ turn 加权成功占比换算表 + 真实 reward 分布
python scripts/rl_tutorial/ch3_reward_to_success.py --sample

# 自定义：如果 critic/score/mean = 0.62，成功轨迹的 turn 占比是多少？
python scripts/rl_tutorial/ch3_reward_to_success.py --reward 0.62

# 验证「平移不变」：把失败 reward 从 -0.1 改成 0，看 advantage 变不变
python scripts/rl_tutorial/ch4_compute_advantage.py --sample --shift 0.1
```

---

## 3.7 这一章你应该带走的

- ✅ **ALFWorld reward 是环境内置的稀疏终止分**：成功 1.0 / 失败 -0.1，不用写 verifier。
- ✅ **终止 reward 广播到每一步**：同一条轨迹的每个 experience，`reward` 都相同（GRPO 用它做组内比较，第 4 章）。
- ✅ **换算前先看统计单位**：`p = (reward + 0.1) / 1.1`；对 `critic/score/mean` 得到的是 turn 加权占比，不是每局等权成功率。
- ✅ **稀疏 reward 靠 G=16 救**：组内成功/失败混合 → 非零 advantage；G=1 时 advantage 恒 0 学不动。
- ✅ **`-0.1` 不改变 advantage**：给组内加常数不改变比较结果，负号只是让曲线更可读（第 4 章展开）。

❌ **你还不需要懂**：

- advantage 的具体公式、怎么广播到 token、PPO clip（→ 第 4 章）
- 更细的「每步功劳分配」方法 GiGPO（→ 第 4 章 §4.3 进阶注脚）
- `reward_std=0` 的「死组」对训练信号的影响（→ 第 4 章）

💡 **留给自己的问题**：你现在知道一个 task 组里有 3 条成功（reward 1.0）、13 条失败（reward -0.1）。GRPO 说「成功的鼓励、失败的抑制」——但**具体鼓励多少、抑制多少**？那 3 条成功轨迹的 advantage 是 +2.0 还是 +0.5？13 条失败的是 -0.46 还是 -2.0？为什么**成功的力度比失败大 4 倍**反而是对的？第 4 章揭晓。

---

**上一章**：[第 2 章：单条 trajectory 内部](./ch2_rollout与experience.md) ｜ **下一章**：[第 4 章：GRPO / GiGPO advantage](./ch4_advantage怎么来.md)
