"""Build the self-contained submission notebook from the reviewed modules."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parent
source = ROOT / 'rl_assignment'
def code_file(name):
    text = (source / name).read_text()
    text = '\n'.join(line for line in text.splitlines()
                     if not line.startswith('from .'))
    return text.split("if __name__ == '__main__':")[0].rstrip()

cells = []
def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))
def code(text):
    cells.append(nbf.v4.new_code_cell(text))

md('''# AI6101 — Q-learning for CliffBoxPushing
**Name:** YourName · **Matriculation number:** YourMatriculationNumber

This notebook is self-contained: run all cells in order. It completes the RLAgent,
value/policy visualization and action-noise evaluation tasks. The supplied environment
below is unchanged. The full experiment trains **9 independent policies** (3 training
noise levels × 3 seeds), for **30,000 episodes each**, then evaluates every policy for
**300 episodes at each test noise**. Expect a few minutes on a CPU; no GPU is required.

Dependencies: Python 3.12, NumPy 2.5.4, Matplotlib 3.11.2, IPython and ipykernel.
All results are computed, not manually entered. The optional GIF uses Pillow.
The companion PDF is exactly two pages; the extended explanation is separate.''')
code('''from pathlib import Path
import copy
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Image
%matplotlib inline

OUTPUT_DIR = Path('results')
OUTPUT_DIR.mkdir(exist_ok=True)
''')
md(r'''## 1. Original environment and MDP
Coordinates use `(row, column)`. State is `(agent_row, agent_col, box_row, box_col)`;
actions 1–4 mean up, down, left and right. There are $84^2=7,056$ Cartesian states
and $28,224$ table entries, including unreachable/terminal states.

The unchanged reward after a transition is
$r=-1-d_1(A,B)-d_1(B,G)-1000\,\mathbb{1}_{cliff}+1000\,\mathbb{1}_{goal}$.
An episode ends at a cliff, the goal, or 100 steps. The supplied implementation,
including its special hard boundary for the box, is authoritative.''')
code((source / 'environment.py').read_text())
md(r'''## 2. Q-learning and action noise
The training policy is ε-greedy with ε=0.1 and uniform random greedy tie-breaking.
With α=0.1 and γ=0.99, update the **intended** action $a$:

$$Q(s,a)\leftarrow Q(s,a)+\alpha\left[r+\gamma(1-d)\max_bQ(s',b)-Q(s,a)\right].$$

Here $d$ is the environment's terminal flag (including the time limit). We use zero
bootstrap at termination. Noise belongs to the transition channel: with probability
$p$ execute a uniformly sampled action from the **other three**; otherwise execute $a$.
Updating the corrupted action instead would learn a different decision problem.

During evaluation ε exploration is disabled, no Q-values change, and only $p_{test}$
noise remains. A separate seeded generator keeps evaluation from consuming the
training generator's random stream. Success means **box at goal**, not positive return.

The four-coordinate state follows the template; it omits remaining time. Thus the
stationary Q-table approximates the capped task rather than an exact time-augmented
finite-horizon MDP. Fixed learning/exploration rates do not establish convergence.''')
code(code_file('agent.py'))
md('''### Small correctness checks
These assertions check the terminal mask and exact noise definition. The repository's
six pytest tests additionally cover nonterminal targets, uniform noise frequencies,
intended-action credit, evaluation isolation, success detection and original dynamics.''')
code('''check_agent = RLAgent(CliffBoxGridWorld(), 0, alpha=0.5, gamma=0.9)
s, next_s = (4, 1, 1, 1), (3, 1, 1, 1)
check_agent.q_table[next_s][1] = 10
check_agent.update(s, 2, -2, next_s, False)
assert np.isclose(check_agent.q_table[s][2], 3.5)
check_agent.update(s, 3, -2, next_s, True)
assert np.isclose(check_agent.q_table[s][3], -1)
rng_check = np.random.default_rng(99)
assert all(apply_action_noise(1, [1,2,3,4], 1, rng_check) != 1 for _ in range(500))
assert all(apply_action_noise(1, [1,2,3,4], 0, rng_check) == 1 for _ in range(100))
print('Notebook correctness checks passed.')''')
md('''## 3. Reproducible experiment
Train noise levels are 0, 0.1, 0.2 with independent seeds 7, 17, 27. All use the same
hyperparameters and 30,000 episodes; there is no reward shaping, action masking,
planning, or map modification. For each trained policy, run 300 episodes for each
test noise: **8,100 evaluation episodes**, plus 1,000 uniform-random baseline episodes.
The seed-7 policies are saved for visualization; they are not selected by evaluation.

Curves use a 500-episode moving average; shading and matrix ± values are sample SD
across the three training seeds, not confidence intervals. Each matrix cell pools
900 evaluation episodes. Raw episode records and per-seed CSV results are saved.''')
code(code_file('experiments.py'))
code('''study = run_study(output_dir=OUTPUT_DIR, episodes=30000,
                  eval_episodes=300, seeds=(7,17,27))
trained_agents = study['agents']  # representative seed-7 policies
agent = trained_agents[0.0]
noise_levels = [0.0, 0.1, 0.2]
success_rate = study['arrays']['success'].mean(axis=0)
average_return = study['arrays']['mean_return'].mean(axis=0)''')
md(r'''## 4. Learning, robustness, final V-table and policy
$V(s)=\max_a Q(s,a)$ is an estimate of the discounted greedy action value; do not
divide it by the number of states. The policy contains all maximizing actions.
Policy arrows depend on **both agent and box** positions. Four slices fix the box
at distinct stages of a successful route; the all-state plots cover the full
agent-index × box-index Cartesian table. Gray means unvisited, not zero value or
safe terrain. A dot denotes a tie. Some visited states still have untried actions
at optimistic zero initialization, so these plots are not a claim of global convergence.''')
code(code_file('plotting.py'))
code('''figures = make_figures(study, OUTPUT_DIR, animation=True)
for figure in figures:
    display(figure)
    plt.close(figure)''')
md('''## 5. Numerical interpretation and a complete greedy trajectory
The noisy environment can terminate a long safe trajectory after a single action
substitution. High clean success therefore need not imply high noisy success.
Training success includes ε exploration as well as noise; evaluation removes ε,
so the two rates measure different policies. Compare a given **column** when
assessing whether training noise helps at a fixed test noise. Three seeds support
a descriptive comparison, not a broad statistical guarantee.''')
code('''print('Success mean ± SD (%):')
print(np.round(100 * success_rate, 2))
print(np.round(100 * study['arrays']['success'].std(axis=0, ddof=1), 2))
print('Mean return ± SD:')
print(np.round(average_return, 2))
print(np.round(study['arrays']['mean_return'].std(axis=0, ddof=1), 2))
baseline = study['summary']['baseline']
print(f"Random baseline: {np.mean(baseline['success']):.1%} success, "
      f"{np.mean(baseline['returns']):.2f} mean return over 1,000 episodes")
rollout = study['summary']['rollout']
print('Clean seed-7 greedy rollout:', len(rollout['actions']), 'steps, return',
      sum(rollout['rewards']), 'success', rollout['success'])
print(' t | state (agent row,col, box row,col) | action | reward')
for t, (state, action, reward) in enumerate(zip(rollout['states'], rollout['actions'], rollout['rewards'])):
    print(f'{t:2d} | {str(state):34s} | {action:6d} | {reward:6.0f}')
display(Image(filename=str(OUTPUT_DIR / 'greedy_rollout.gif'), format='gif'))''')
md('''### Findings from the recorded experiment
Clean-trained policies achieve 100% clean-test success, 48.22% at test noise 0.1 and
18.56% at 0.2 (means across three seeds). Both noisy-training settings have zero
successes across all three test noises. Each noisy-training setting also has zero
goal visits in its 90,000 training episodes across seeds; all clean greedy evaluations
of those policies time out. This is consistent with failure to discover successful
experience under the chosen training budget and fixed exploration rate. It does not
establish that noisy training is always ineffective. These are recorded-run observations,
not values used by the implementation. The matrices and figures above are computed
directly by the code.''')
md('''## 6. Limitations and reproducibility
The deterministic clean task can give identical greedy outcomes across hundreds of
evaluation episodes. Those repetitions do not demonstrate independent training
reliability; the three independent training seeds address that separately. No claim
is made that the learned route is shortest or globally optimal. Failure under noise
is retained and discussed rather than discarded. We did not attempt the optional
UCB or sparse-reward bonus tasks.

References: Watkins & Dayan (1992), *Q-learning*, Machine Learning 8, 279–292;
Sutton & Barto (2018), *Reinforcement Learning: An Introduction*, 2nd ed., §6.5;
AI6101 assignment brief and supplied `RLAgent_2026Fall.ipynb` environment.''')

notebook = nbf.v4.new_notebook(cells=cells, metadata={
    'kernelspec': {'display_name': 'Python (AI6101 RL)', 'language': 'python', 'name': 'ai6101'},
    'language_info': {'name': 'python', 'version': '3.12.14'}})
path = ROOT / 'deliverables/AI6101_Code_YourName_YourMatriculationNumber.ipynb'
nbf.write(notebook, path)
print(path)
