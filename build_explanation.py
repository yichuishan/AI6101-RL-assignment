"""Generate the extended Chinese explanation, separate from the two-page report."""
from pathlib import Path
import json
import numpy as np

root = Path(__file__).resolve().parent
s = json.loads((root / 'results/summary.json').read_text())
a = np.load(root / 'results/experiment_arrays.npz')
success = a['success'].mean(0)
returns = a['mean_return'].mean(0)
success_sd = a['success'].std(0, ddof=1)
return_sd = a['mean_return'].std(0, ddof=1)

def table(values, std, percentage=False):
    rows = ['| 训练噪声 / 测试噪声 | 0 | 0.1 | 0.2 |', '|---|---:|---:|---:|']
    scale = 100 if percentage else 1
    for i, p in enumerate([0, .1, .2]):
        cells = [f'{values[i,j]*scale:.2f} ± {std[i,j]*scale:.2f}' for j in range(3)]
        rows.append('| '+str(p)+' | '+' | '.join(cells)+' |')
    return '\n'.join(rows)

rollout=s['rollout']
delta=100*(success[1]-success[0])
text = r'''# AI6101 代码逻辑、强化学习知识与实验解读

本文是扩展讲解，不属于固定两页的英文报告。数值来自实际运行的实验；图表与 Notebook、PDF 使用同一份结果。

## 1. 文件怎么使用

- `AI6101_Code_YourName_YourMatriculationNumber.ipynb`：自包含、已执行的提交代码。所有算法、原始环境、评估和绘图均写在 Notebook 中；提交时不依赖旁边的 Python 模块或结果文件。
- `AI6101_Report_YourName_YourMatriculationNumber.pdf`：严格两页的英文报告。
- `AI6101_Project_YourName_YourMatriculationNumber.zip`：只包含上述两个正式提交文件。
- 本文及 `results/`：用于理解、复核和复现，不计入报告页数，也不放入双文件提交 ZIP。

姓名和学号当前使用占位符。提交前请同步替换 Notebook 首页、报告姓名栏以及三个提交文件名；报告应从修改后的 `build_report.py` 重新生成。

## 2. 这个环境到底在学什么

网格有 6 行、14 列，坐标从零开始。人物初始位置为 `(4,1)`，箱子为 `(1,1)`，目标为 `(4,13)`。动作 `1/2/3/4` 分别对应上、下、左、右。人物走到箱子旁边并朝箱子移动时，才可能推箱子；人物不能拉箱子。因此，仅仅让人物或箱子靠近目标不够，还必须保留下一步能够推动箱子的站位。

### 状态为什么需要四个数

状态为 `s=(人物行, 人物列, 箱子行, 箱子列)`。同一个人物位置，在不同箱子位置下应采取不同动作。例如，要向上推箱子，人物必须先走到箱子下方。因此，只以人物位置为索引的 6×14 Q 表会丢失关键信息。

四维笛卡尔表有 `(6×14)²=7,056` 个状态，每个状态有 4 个动作，共 `28,224` 个 Q 值。这只是存储空间：人物与箱子重叠、某些边界或悬崖状态并不是正常可继续决策的状态；不是每个表项都会被访问。

### 奖励、成功与终止

原始环境在一次动作之后计算奖励：

$$r=-1-d_1(A,B)-d_1(B,G)-1000\mathbf{1}_{cliff}+1000\mathbf{1}_{goal},$$

其中 $d_1$ 是曼哈顿距离。人物或箱子掉入悬崖、箱子到达目标、步数到 100，任一条件成立就结束回合。最终到达目标的一步仍包含基础扣分与距离扣分；正常相邻推动到目标时，最后一步奖励为 `1000−1−1=998`，不是恰好 1000。

**成功率按箱子是否真的到达目标统计。** `回报 > 0` 不是可靠的成功定义：成功之前可能已经累计很多负奖励。代码始终检查 `np.array_equal(env.box_pos, env.goal_pos)`。

环境保留了模板的特殊箱子边界：箱子不能被推到第 0 行、第 5 行和第 0 列；人物仍能在外侧行列绕行。不能把它当成所有对象都使用完全相同边界的普通走格子游戏。地图、奖励函数和 `step()` 均未改变。

## 3. Q-learning 的核心知识

### Q 值与 V 值

$Q(s,a)$ 表示在状态 $s$ 选择动作 $a$ 后，未来折扣回报的估计。它考虑后续收益，而不只看眼前的一步奖励。与之对应，贪心状态价值估计是

$$V(s)=\max_a Q(s,a),\qquad \pi(s)\in\arg\max_a Q(s,a).$$

这里的 $V(s)$ 不需要除以状态数。作业模板打印部分的 `val / state_num` 不代表标准状态价值，因此实现直接展示原始 `max Q`。如果多个动作并列最大，策略保留全部并列动作；执行时均匀随机选一个，避免固定动作顺序造成偏置。

### 更新公式拆开看

$$y=r+\gamma(1-d)\max_b Q(s',b),$$
$$Q(s,a)\leftarrow Q(s,a)+\alpha\big[y-Q(s,a)\big].$$

- `r`：本次实际得到的奖励。
- `γ=0.99`：折扣因子，控制未来收益的重要程度。
- `max Q(s′,·)`：对下一状态未来最优收益的当前估计。
- `d`：是否终止，结束后不再估计后续价值。
- `α=0.1`：学习率，每次只向目标移动一部分。
- `y−Q(s,a)`：时间差分误差，简称 TD error。

例如，令 `Q(s,a)=0`、`r=−2`、下一状态最大 Q 值为 `10`，测试中取 `α=0.5, γ=0.9`：非终止时目标是 `−2+0.9×10=7`，更新后为 `3.5`；如果这一步已经终止，目标只能是 `−2`，更新后为 `−1`。终止掩码错误会把不存在的未来收益传播回去，测试专门检查了这一点。

### 为什么这是 off-policy 方法

训练行为是 ε-greedy：以 `ε=0.1` 的概率随机尝试四个动作，否则选择当前 Q 值最大的动作；但更新目标使用下一状态的 `max Q`，而不是训练行为随后实际抽到的动作。因此行为策略和目标策略不同，Q-learning 属于 off-policy 方法。与之相比，SARSA 使用实际下一动作 `Q(s′,a′)`，对持续探索下的路径风险可能有不同表现；本项目没有运行 SARSA，因此不声称它优于或劣于本实现。

全零初始化在负的逐步奖励下带有乐观性：未尝试动作可能暂时比已经尝试过的负值动作更吸引人。它有利于早期探索，也意味着一些罕见状态的 `max Q` 仍可能被未尝试动作的零值主导。

## 4. 动作噪声最容易写错的地方

### 不是再随机一次四个动作

题目规定：以概率 `p` 把原动作替换成**另外三个动作之一**，每个概率为 `p/3`。因此

$$P(\tilde a=a\mid a)=1-p,\qquad P(\tilde a=b\ne a\mid a)=p/3.$$

如果在触发噪声后又从全部四个动作抽取，原动作仍可能被抽回，实际替换率会变成 `3p/4`，不符合题意。`apply_action_noise()` 先排除原动作，再均匀抽样。测试同时验证 `p=0`、`p=1`、替代动作的均匀性和 `p=0.2` 的经验替换频率。

### 更新 intended action，而不是 executed action

算法先决定 `intended`，噪声把它变成 `executed`，环境执行后返回奖励。Q 表学习的是“我选择某个指令，经噪声通道执行后的期望结果”，因此更新 `Q(s,intended)`：

```text
state ──ε-greedy──> intended action ──noise(p)──> executed action
   ▲                                              │
   │                                              ▼
   └──── state ← next_state      original env.step(executed)
                                  │ reward, next_state, done
                                  ▼
                     update Q(state, intended action)
```

把更新记在 `executed` 上，会学到“指令没有失真时各实际动作的结果”，而不是带噪声控制问题。本实现有单步人工环境测试：强制把动作 1 换掉，仍然只更新动作 1 的 Q 值。

### ε 探索和动作噪声不是一回事

训练时两者都存在；测试时关闭 ε，只保留指定测试噪声。假设唯一贪心动作，`ε=0.1` 时，随机探索仍有 1/4 概率碰巧选回贪心动作，因此训练中实际执行非贪心动作的概率为 `0.075 + 0.9p`，分别约为 `7.5%、16.5%、25.5%`。这个推导只适用于唯一贪心动作，不能直接当成所有状态的统计结果。

测试时即使没有 ε，并列最大 Q 值仍会随机打破平局；这属于贪心策略的定义，不是额外 ε 探索。评估使用单独的随机数生成器，既不消耗训练 RNG，也不修改 Q 表。

## 5. 代码按什么顺序运行

1. **原始环境**：直接放入提供的环境代码，不重写转移规则。
2. **`RLAgent.__init__()`**：建立四维状态字典、每个动作的 Q 值、访问计数和独立 RNG。
3. **`act()`**：训练时 ε-greedy，测试时纯贪心；最大值并列时随机选择。
4. **`learn()`**：每回合重置；选动作、加噪声、调用 `env.step()`、更新 Q，直到回合结束；记录回报、真实成功标志与长度。
5. **`evaluate_details()` / `evaluate()`**：新建同样的环境，关闭探索且不更新参数，记录成功、平均回报、悬崖失败率和超时率。`evaluate()` 保留模板的 `(success_rate, mean_return)` 接口。
6. **`run_study()`**：三个种子、三种训练噪声分别训练九张表，再完成全部九种训练/测试组合；保存原始数据与模型。
7. **`visualize()` / `make_figures()`**：计算 V 表、贪心策略、学习曲线、热力图、全状态可视化、轨迹与 GIF。

Notebook 中完整包含上述代码。仓库里的 `rl_assignment/` 模块是便于复核、测试和维护的同一份实现；不是提交 Notebook 的运行依赖。

## 6. 实验设置和真实结果

固定使用 `α=0.1, γ=0.99, ε=0.1`。每个策略训练 **30,000 回合**，训练种子为 **7、17、27**，训练噪声为 **0、0.1、0.2**，共九次独立训练。每个训练好的策略在三种测试噪声下各评估 **300 回合**，总计 **8,100 次评估**；每个热力图格子对应三个种子各 300 回合，即合计 900 回合。随机基线另跑 1,000 回合。

这里的 “±” 是**三个训练种子的指标之间的样本标准差**，不是 900 回合的标准差，也不是置信区间。只有三个训练种子，不宜作强统计显著性结论。无噪声环境中的重复评估可能走完全相同路线，因此 300 次成功不代表进行了 300 次独立训练。

### 6.1 学习曲线

![学习曲线](../results/learning_curves.png)

左图是未折扣 episode return，右图是训练回合真正到达目标的比例。线条先对每个种子做 500 回合滑动平均，再跨种子求平均；阴影为这些平滑曲线的 ±1 标准差。横轴 500 之前没有完整窗口，故从第 500 回合开始。原始逐回合数据仍保存在 NPZ 中，不是只保存平滑后的曲线。

Q 学习目标使用折扣收益，而作业评价曲线是未折扣回报，两者用途不同。训练曲线包含 ε 探索，测试不含 ε，因此训练期成功率明显低于最终干净测试成功率不一定是程序问题。

### 6.2 成功率矩阵（百分比）

SUCCESS_TABLE

### 6.3 平均回报矩阵（未折扣）

RETURN_TABLE

![噪声鲁棒性热力图](../results/robustness_heatmaps.png)

判断“训练时加噪声是否有帮助”，必须固定测试噪声，比较**同一列**，不能把干净测试和带噪声测试混在一起。训练噪声 0.1 相对干净训练，在测试噪声 0、0.1、0.2 下的成功率变化分别为 **DELTA_POINTS 个百分点**。这些数值是本次固定训练预算下的观察，不是普遍规律，也不应该为了让结果更好看而删除失败种子。

本次的具体诊断是：训练噪声为 **0.1 和 0.2 时，每种设置跨三个种子合计 90,000 个训练回合，都没有到达过目标**。它们在干净贪心测试中也全部超时，而不是成功到达目标。在带噪声测试中，则同时出现掉崖与超时，比例记录在 `evaluation_by_seed.csv` 中。因为从未得到最终目标的正奖励，目标收益没有通过成功轨迹向前传播；阶段性回报改善不能自动解释为已经学会推箱任务。

这些证据与“在固定探索率和训练预算下没有发现有效成功经验”一致；它们不证明带噪声问题无解，也不证明噪声训练永远无效。没有额外运行更长训练、课程学习或调整超参数的消融，因此不能断言哪一种改进一定奏效。题目明确说明，某些方法在噪声下可能难以收敛，低成功率甚至零成功率本身可以是有效实验结果。

动作噪声既会改变状态分布，也会让“在危险位置附近获得成功经验”更难。干净策略即使很可靠，也可能非常脆弱：一次错误动作就可能掉崖；适量噪声训练则可能学到不同的风险取舍，也可能因学习难度增大而变差。结合两张矩阵比较：成功率描述能否完成任务，平均回报还受到死亡前消耗步数、人物与箱子距离、箱子与目标距离的影响，二者并不严格一一对应。

随机基线：**BASELINE_RATE 成功率，平均回报 BASELINE_RETURN**（1,000 回合）。

## 7. 最终 V 表、策略和轨迹怎么读

![状态价值与策略切片](../results/value_policy_slices.png)

四个面板分别固定箱子在 `(1,1)、(4,5)、(1,10)、(2,13)`，再让人物位置遍历整个地图。底色是该联合状态的 `max Q`；箭头是唯一的贪心动作；点表示存在并列最大动作；深灰叉号是悬崖；浅灰是未访问状态；橙色 B 是固定的箱子，绿色 G 是目标。人物所在格子的箭头并不代表箱子的运动方向。每个面板对应的是不同的箱子位置，因此不能把箭头拼成只依赖人物坐标的统一策略。

![完整联合状态表](../results/all_state_value_policy.png)

全状态图用 `人物索引=14×行+列` 和 `箱子索引=14×行+列` 展示全部 7,056 个表项。该图完整但不便直接导航；切片图更适合解释具体决策。未访问状态没有足够证据给出可靠动作，所以主动遮罩，而不是把全零初值的任意箭头当成已经学到的策略。已经访问过某状态，也不代表它的四个动作都被充分学习。

![人物和箱子轨迹](../results/greedy_trajectory.png)

![逐步推箱子动画](../results/greedy_rollout.gif)

图和动画固定展示**预先指定的种子 7、训练噪声 0 的策略**，没有从九个策略中按结果挑选最好的一次。该干净贪心回合为 **ROLLOUT_STEPS 步，总回报 ROLLOUT_RETURN，成功标志 ROLLOUT_SUCCESS**。Notebook 还输出每一步的状态、动作、奖励，便于逐步对照环境逻辑。该路线是实际学习结果；没有声称它是最短或全局最优路线。

## 8. 已经验证了什么

- 13 项 pytest 测试：原有六项验证之外，新增 UCB 探索计数、无探索训练、严格 0/1000 奖励、安全动作选项真实执行、半马尔可夫折扣、稀疏评估冻结 Q 值，以及成功回放不虚构奖励的检查。
- 提交 Notebook 从头到尾自动执行，保留真实输出与图表，不需要手工输入动作。
- 原始环境代码与上传 Notebook 的环境代码逐字核对。
- 论文 PDF 的页数与版面经过检查，固定为两页。
- 评估每种设置都超过要求的 100 回合；所有失败均计入统计，未进行筛选。

测试通过说明这些关键实现与评估规则符合预期，并不能证明算法必然找到全局最优解。

## 9. 方法限制与可以进一步研究的方向

1. **时间没有进入状态。** 模板只使用人物和箱子四个坐标，但 100 步上限意味着同样位置在剩余 1 步和剩余 90 步时的价值不同。这里遵循模板，用同一张静态 Q 表近似有限时域问题，并在时间终止时关闭后续估计；没有宣称它是包含剩余时间的精确最优有限时域解。
2. **固定学习率和探索率。** 本实验强调公平、可复现比较，使用所有设置共用的固定超参数；没有满足所有经典 Q-learning 渐近收敛定理的条件，所以不能用这些定理声称已经收敛。
3. **三个种子的统计能力有限。** 对噪声训练的改善或退化作描述性分析，不把几个百分点的变化自动解释为显著优势。
4. **先验不同，需要分别解释。** 必做实验和探索策略比较未加入安全规则或规划；稀疏奖励层次方法使用基于地图的安全站位 BFS 和一次推动选项。这是明确披露的技能先验，通过随机选项及无回放消融来区分其与学习的作用，不能把全部提升都归因于原始 Q-learning。
5. **额外计算并非免费。** 成功回放不增加环境交互，但增加 TD 更新次数；相同回合数和 100 步上限也不意味着实际交互数完全相同。新增实验逐种子记录这些开销。

可以继续探索：带时间状态的有限时域 Q 学习、SARSA 与 Q-learning 的风险差异、探索率退火、更多训练种子与训练预算、置信区间和风险敏感目标。这些都属于后续实验，本次结果不包含这些方法。

## 10. 本地复现

在当前云环境中：

```bash
cd /workspace/AI6101-RL-assignment
source /workspace/.cloud-setup/ai6101/env.sh
python -m pytest -q tests
# 单独重新训练并输出原始结果（会覆盖 results 中本次同名实验文件）：
python -m rl_assignment.experiments
# 独立运行两个加分实验（复用已有的干净 ε-greedy 对照）：
python -m rl_assignment.bonus
```

更直接的使用方式是在 Jupyter 中打开提交 Notebook，选择 AI6101 内核后按顺序 Run All。它不依赖源码目录，输出会写到当前工作目录的 `results/`。必做实验是九个策略、每个 30,000 回合。第 7、8 节在结果文件存在时复用已完成的加分实验，设置 `RECOMPUTE_BONUS=True` 可强制重新训练；单独下载 Notebook、没有结果文件时会自动训练。修改算法或参数后，应删除缓存或强制重跑。如果只是阅读，查看已经保存的输出即可。

更换机器时，可安装 `requirements.txt` 中固定版本的 NumPy、Matplotlib、JupyterLab、ipykernel 与 pytest。生成 PDF 还需要 `requirements-report.txt`；这与运行强化学习 Notebook 是两组用途不同的依赖。

## 11. 参考资料

- Watkins, C. J. C. H. & Dayan, P. (1992). Q-learning. *Machine Learning*, 8, 279–292. DOI: 10.1007/BF00992698。
- Sutton, R. S. & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*, 2nd edition, §6.5。
- 本次上传的 AI6101 作业说明及 `RLAgent_2026Fall.ipynb` 原始环境。
- Auer, Cesa-Bianchi & Fischer (2002). *Finite-time Analysis of the Multiarmed Bandit Problem*。
- Sutton, Precup & Singh (1999). *Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning*。
'''
text=text.replace('SUCCESS_TABLE', table(success, success_sd, True))
text=text.replace('RETURN_TABLE', table(returns, return_sd))
text=text.replace('DELTA_POINTS', '、'.join(f'{x:+.2f}' for x in delta))
text=text.replace('BASELINE_RATE', f'{np.mean(s["baseline"]["success"]):.2%}')
text=text.replace('BASELINE_RETURN', f'{np.mean(s["baseline"]["returns"]):.2f}')
text=text.replace('ROLLOUT_STEPS', str(len(rollout['actions'])))
text=text.replace('ROLLOUT_RETURN', f'{sum(rollout["rewards"]):.0f}')
text=text.replace('ROLLOUT_SUCCESS', str(rollout['success']))
ex=json.loads((root/'results/bonus_exploration.json').read_text())
sp=json.loads((root/'results/bonus_sparse.json').read_text())
ex_table=['| 方法 | 首次成功：种子 7 / 17 / 27 | 最终成功率 | 平均回报 |','|---|---|---:|---:|']
for method in ex['methods']:
    rows=[r for r in ex['rows'] if r['method']==method]
    first=' / '.join(str(r['first_success']) for r in rows)
    ex_table.append(f"| {method} | {first} | {np.mean([r['eval_success'] for r in rows]):.1%} | {np.mean([r['eval_return'] for r in rows]):.2f} |")
sp_table=['| 方法 | 首次成功：种子 7 / 17 / 27 | 最终成功率 | 平均稀疏回报 | 平均测试步数 |','|---|---|---:|---:|---:|']
cost_table=['| 方法 | 平均环境交互数 | 平均额外回放更新数 |','|---|---:|---:|']
for method in sp['methods']:
    rows=[r for r in sp['rows'] if r['method']==method]
    first=' / '.join('未成功' if r['first_success'] is None else str(r['first_success']) for r in rows)
    sp_table.append(f"| {method} | {first} | {np.mean([r['eval_success'] for r in rows]):.1%} | {np.mean([r['eval_return'] for r in rows]):.1f} | {np.mean([r['eval_steps'] for r in rows]):.1f} |")
    cost_table.append(f"| {method} | {np.mean([r['total_interactions'] for r in rows]):,.0f} | {np.mean([r['replay_updates'] for r in rows]):,.0f} |")
bonus=r'''## 新增第 7 节：探索策略比较（第一个加分任务）

对照为默认 ε-greedy、UCB、训练阶段 ε=0 的贪心策略。均使用原始奖励、原始动作、不加动作噪声；学习率 0.1、折扣 0.99、每种策略训练 30,000 回合，种子 7/17/27。ε-greedy 复用必做实验中完全相同的干净训练结果。每种策略每个种子评估 300 回合，关闭全部探索加成。

UCB 先尝试未尝试的动作，再最大化 `Q(s,a) + 20 * sqrt(log(N(s)+1) / N(s,a))`。计数加成只用于训练选动作，不进入奖励或 TD 目标。系数 20 使用原始奖励单位；这个 Q-learning 版本是探索启发式，不直接提供严格统计置信区间保证。

EX_TABLE

首次成功中位回合数为 **8,038、5,106、4,735**。UCB 早于 ε-greedy，却没有超过贪心对照；三者最终均达到 100% 干净测试成功率。原始负逐步奖励使全零初始化具有乐观性：未尝试动作的零值可能高于已尝试动作的负值；随机平局选择也提供探索。所以“ε=0”不等于“没有任何探索行为”，本实验不支持 UCB 普遍优胜的结论。

## 新增第 8 节：纯稀疏奖励与层次化学习（第二个加分任务）

只使用 **箱子到达原目标奖励 1000，否则 0**。丢弃原始密集奖励；距离变化、掉崖、超时均不额外加减分。地图、初始位置和每回合 100 个原始动作的上限不变，没有密集预训练、示范、内在奖励或目标重标注。

安全推动选项是手工技能：BFS 只寻找人物绕到箱子所需一侧的安全路径，再执行一次安全推动。不规划完整箱子到目标路线，也不根据目标距离挑选方向；高层 Q-learning 学习选择下一次推动方向。**这是基于地图知识的技能/安全先验**，报告明确披露，并通过随机选项、不带回放的选项学习进行消融。

每个选项消耗 k 个真实 `env.step()` 动作，全部计入 100 步上限。其目标为：

$$y=\sum_{i=0}^{k-1}\gamma^i r_i+(1-d)\gamma^k\max_o Q(s',o).$$

下一选项价值必须使用 `gamma**k`；如果成功发生在选项末步，其选项内奖励为 `1000 * gamma**(k-1)`。达到步数上限时立即按原环境终止。无可行推动时，不提前重置，而用安全的真实人物动作耗尽原回合。

四种方法均训练 5,000 回合/种子，评估 300 回合/种子。学习器测试关闭 ε 和更新；随机选项基线仍按定义随机选择。

SP_TABLE

只有**选项 Q＋成功回放**在本次最终测试中达到 **900/900 成功**，平均稀疏回报为 1000。满足 `平均回报 = 1000 × 成功率`。种子 7 的成功路线为 45 步，另两种子为 39、43 步；轨迹在 `bonus_sparse_rollout.json`，未声称全局最短。

回放只在真实成功后进行，将实际选项转移从后向前额外更新 10 遍。没有成功时回放次数为零，不虚构奖励。带回放与不带回放的首次成功回合完全相同（1,155 / 3,454 / 2,394），之后结果才分开：回放改善**成功信息的传播与利用**，首次发现来自探索与技能先验。随机技能和无回放学习在训练中曾偶尔成功，最终测试却没有成功，说明找到一条成功经历不等于形成可靠策略。

![两个加分实验的学习曲线](../results/bonus_learning_curves.png)

阴影为三个训练种子的样本标准差。右图没有加入任何中间奖励。

COST_TABLE

相同回合数和上限不意味着实际交互相同：随机原始动作容易提前掉崖，安全选项经常运行到超时，学会后又会提前成功。回放不增加环境交互，却增加 TD 计算。因此记录这些开销，不把额外计算视为免费，也不把安全技能带来的全部改善都归因于 Q-learning。

'''
bonus=bonus.replace('EX_TABLE','\n'.join(ex_table)).replace('SP_TABLE','\n'.join(sp_table)).replace('COST_TABLE','\n'.join(cost_table))
text=text.replace('## 8. 已经验证了什么',bonus+'## 8. 已经验证了什么')
path=root/'deliverables/代码逻辑与实验详解.md'
path.write_text(text)
print(path)
