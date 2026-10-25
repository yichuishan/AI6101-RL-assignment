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

md(r'''# AI6101 — Q-learning for CliffBoxPushing
**Name:** YourName · **Matriculation number:** YourMatriculationNumber

This project trains a tabular Q-learning agent to push the box to the goal. The experiments examine learning progress, the learned value function and policy, action-noise robustness, exploration strategies, and hierarchical learning with sparse rewards.''')
code('''from pathlib import Path
import copy
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Image
%matplotlib inline

OUTPUT_DIR = Path('results')
OUTPUT_DIR.mkdir(exist_ok=True)
''')
md(r'''## 1. Grid World and State Representation
The environment contains 6 rows and 14 columns. The agent starts at `(4,1)`, the box at `(1,1)`, and the goal is `(4,13)`. Actions 1–4 correspond to up, down, left, and right.

Each state contains both positions:

$$s=(a_r,a_c,b_r,b_c).$$

The Q-table contains $84^2=7,056$ Cartesian states and four actions per state, giving 28,224 entries.

After each transition, the original reward is

$$r=-1-d_1(A,B)-d_1(B,G)-1000\mathbb{1}_{\mathrm{cliff}}+1000\mathbb{1}_{\mathrm{goal}},$$

where $d_1$ is Manhattan distance. An episode ends when the agent or box enters a cliff, the box reaches the goal, or 100 primitive steps have been executed. The box is restricted to rows 1–4 and columns 1–13; the agent can use the outer rows and columns to reposition.''')
code((source / 'environment.py').read_text())
md(r'''## 2. Q-learning and Action Noise
Q-values are initialized to zero. Training uses ε-greedy action selection with ε=0.1, learning rate α=0.1, and discount factor γ=0.99. Ties between maximizing actions are resolved uniformly at random.

The update is

$$Q(s,a)\leftarrow Q(s,a)+\alpha\left[r+\gamma(1-d)\max_bQ(s',b)-Q(s,a)\right],$$

where $d$ is the terminal flag. Terminal transitions use the immediate reward as the target.

With probability $p$, the selected action is replaced uniformly by one of the other three actions. The environment executes the resulting action, and Q-learning updates the entry for the intended action. This learns the expected outcome of sending each action through the noisy transition channel.

Evaluation uses greedy action selection, fixed Q-values, and the specified test noise. Success is recorded when the box reaches the goal.''')
code(code_file('agent.py'))
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
md(r'''## 3. Experimental Setup
Agents are trained with action-noise levels $p_{\mathrm{train}}\in\{0,0.1,0.2\}$ and random seeds 7, 17, and 27. Each policy receives 30,000 training episodes with the same Q-learning parameters and original reward function.

Each trained policy is evaluated for 300 episodes at every test-noise level $p_{\mathrm{test}}\in\{0,0.1,0.2\}$. This gives 8,100 evaluation episodes, with 900 episodes contributing to each train/test-noise matrix cell. A uniform-random baseline is evaluated for 1,000 episodes.

Training curves use 500-episode moving averages. Lines show the mean across three seeds, and shaded bands show one sample standard deviation. The evaluation matrices report the mean and sample standard deviation across the three trained policies.''')
code(code_file('experiments.py'))
code('''study = run_study(output_dir=OUTPUT_DIR, episodes=30000,
                  eval_episodes=300, seeds=(7,17,27))
trained_agents = study['agents']  # representative seed-7 policies
agent = trained_agents[0.0]
noise_levels = [0.0, 0.1, 0.2]
success_rate = study['arrays']['success'].mean(axis=0)
average_return = study['arrays']['mean_return'].mean(axis=0)''')
md(r'''## 4. State Values and Learned Policy
The state-value estimate and greedy policy are obtained from the Q-table:

$$V(s)=\max_aQ(s,a),\qquad \pi(s)\in\arg\max_aQ(s,a).$$

Four policy slices fix the box at `(1,1)`, `(4,5)`, `(1,10)`, and `(2,13)` while varying the agent position. Colors show state values, arrows show unique greedy actions, dots show tied actions, and gray cells show unvisited states. B marks the box and G marks the goal.

The complete joint-state plots use agent and box indices defined by $14\times\mathrm{row}+\mathrm{column}$. The highest-valued visited states and their greedy actions are also listed.''')
code(code_file('plotting.py'))
code('''figures = make_figures(study, OUTPUT_DIR, animation=True)
for figure in figures:
    display(figure)
    plt.close(figure)''')
md(r'''## 5. Action-Noise Results and Greedy Trajectory
Clean-trained policies achieve 100% success without test noise. Their success rates decrease to 48.22% at test noise 0.1 and 18.56% at test noise 0.2. Action substitutions near cliffs interrupt trajectories that succeed under deterministic execution.

Agents trained with noise 0.1 and 0.2 record no goal visits across 90,000 training episodes per setting and achieve zero success across the three test-noise levels. Their noise-free greedy evaluations end in timeout. Successful goal-reaching behavior did not emerge under these training settings.

Training success includes ε exploration and action noise, while evaluation removes ε exploration. The seed-7 clean-trained greedy policy completes the task in 39 primitive steps with an undiscounted return of 577. The following trajectory and animation show its actions, states, and rewards.''')
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

md(r'''## 6. Comparison of Exploration Strategies
The exploration study compares ε-greedy training with ε=0.1, UCB action selection, and greedy training with ε=0. All methods use the original reward, primitive actions, zero action noise, α=0.1, γ=0.99, and 30,000 training episodes per seed. The clean ε-greedy runs from the preceding experiment provide the ε-greedy results. Each policy is evaluated for 300 episodes per seed with exploration disabled.

UCB first selects actions that have not been tried in the current state. Once all actions have been sampled, it maximizes

$$Q(s,a)+20\sqrt{\frac{\ln(N(s)+1)}{N(s,a)}},$$

where $N(s,a)$ counts selections of action $a$ in state $s$, and $N(s)=\sum_aN(s,a)$. The exploration term is used for action selection; Q-values are updated using the original Q-learning target. All three strategies use random tie-breaking.''')
bonus_source = code_file('bonus.py')
exploration_source, sparse_source = bonus_source.split('def sparse_reward(env):', 1)
code(exploration_source)
code('''import json
RECOMPUTE_BONUS = False
exploration_files = [OUTPUT_DIR/'bonus_exploration.npz', OUTPUT_DIR/'bonus_exploration.json']
if RECOMPUTE_BONUS or not all(path.exists() for path in exploration_files):
    exploration_bonus = run_exploration_bonus(OUTPUT_DIR, reference_study=study)
else:
    exploration_bonus = {'arrays':dict(np.load(exploration_files[0])),
                         'summary':json.loads(exploration_files[1].read_text())}
    assert exploration_bonus['summary']['episodes'] == 30000
    assert exploration_bonus['summary']['seeds'] == [7,17,27]
    assert exploration_bonus['summary']['ucb_c'] == 20
    print('Reusing completed, recorded exploration experiments; set RECOMPUTE_BONUS=True to retrain.')
print('method | seed | first goal episode | greedy success | mean original return')
for row in exploration_bonus['summary']['rows']:
    print(row['method'], row['seed'], row['first_success'],
          f"{row['eval_success']:.1%}", f"{row['eval_return']:.2f}", sep=' | ')
ev = exploration_bonus['arrays']['evaluation']
for j, method in enumerate(exploration_bonus['summary']['methods']):
    print(f"{method}: greedy success {ev[:,j,0].mean():.1%} "
          f"± {ev[:,j,0].std(ddof=1):.1%} across seeds")''')

md(r'''## 7. Sparse-Reward Learning with Push Options and Replay
The sparse experiment uses reward 1000 when the box reaches the original goal and 0 for every other transition. The map, initial positions, transition rules, and 100-primitive-step episode limit are retained.

The hierarchical learner uses hand-designed safe-push options. For each feasible push direction, local BFS finds a safe path that positions the agent on the required side of the box, followed by one push. The options use the map and box-boundary rules, while the high-level Q-table learns which push direction to select. Every primitive action is executed through the environment and counted toward the episode limit. In states with no feasible push, the agent walks safely until timeout.

For an option lasting $k$ primitive steps, the semi-Markov Q-learning target is

$$y=\sum_{i=0}^{k-1}\gamma^ir_i+(1-d)\gamma^k\max_oQ(s',o).$$

The replay variant performs 10 backward update passes over the option transitions of each successful training episode.

Four methods are compared: primitive ε-greedy Q-learning, random selection of safe-push options, option Q-learning, and option Q-learning with successful-trajectory replay. Each method trains for 5,000 episodes using seeds 7, 17, and 27, followed by 300 evaluation episodes per seed. Learned policies are evaluated greedily; the random-options baseline continues to select options uniformly. Environment interactions and replay updates are recorded for each method.''')
code('def sparse_reward(env):'+sparse_source)
code('''sparse_files = [OUTPUT_DIR/'bonus_sparse.npz', OUTPUT_DIR/'bonus_sparse.json',
                OUTPUT_DIR/'bonus_sparse_rollout.json']
if RECOMPUTE_BONUS or not all(path.exists() for path in sparse_files):
    sparse_bonus = run_sparse_bonus(OUTPUT_DIR)
else:
    sparse_bonus = {'arrays':dict(np.load(sparse_files[0])),
                    'summary':json.loads(sparse_files[1].read_text()),
                    'rollout':json.loads(sparse_files[2].read_text())}
    assert sparse_bonus['summary']['episodes'] == 5000
    assert sparse_bonus['summary']['seeds'] == [7,17,27]
    assert sparse_bonus['summary']['replay_passes'] == 10
    print('Reusing completed, recorded sparse experiments; set RECOMPUTE_BONUS=True to retrain.')
print('method | seed | first success | greedy success | steps | interactions | extra replay updates')
for row in sparse_bonus['summary']['rows']:
    print(row['method'], row['seed'], row['first_success'],
          f"{row['eval_success']:.1%}", f"{row['eval_steps']:.1f}",
          row['total_interactions'], row['replay_updates'], sep=' | ')
ev = sparse_bonus['arrays']['evaluation']
assert np.allclose(ev[:,:,1], 1000*ev[:,:,0])
for j, method in enumerate(sparse_bonus['summary']['methods']):
    print(f"{method}: success={ev[:,j,0].mean():.2%} ±{ev[:,j,0].std(ddof=1):.2%}; "
          f"mean sparse return={ev[:,j,1].mean():.2f}")
figure = plot_bonus_results(exploration_bonus, sparse_bonus, OUTPUT_DIR)
display(figure); plt.close(figure)
print('Seed-7 learned sparse rollout:', len(sparse_bonus['rollout']['actions']), 'steps,',
      sum(sparse_bonus['rollout']['rewards']), 'return, success=',sparse_bonus['rollout']['success'])
print('Reward support:', sorted(set(sparse_bonus['rollout']['rewards'])))
assert set(sparse_bonus['rollout']['rewards']) <= {0.,1000.}
assert len(sparse_bonus['rollout']['actions']) <= 100
print('Sparse reward and primitive-step horizon checks passed.')''')

md(r'''## 8. Discussion of Experimental Results
All three exploration strategies achieved 100% final success under noise-free evaluation. The median first-success episodes were 8,038 for ε-greedy, 5,106 for UCB, and 4,735 for greedy training. UCB accelerated goal discovery relative to ε-greedy, while greedy training discovered successful trajectories earliest. Optimistic zero initialization and random tie-breaking provided exploration even with ε=0: negative updates made previously untried actions more attractive than actions already associated with negative returns.

Under purely sparse rewards, primitive Q-learning, random safe options, and option Q-learning without replay each achieved 0% final evaluation success. Option Q-learning with successful-trajectory replay achieved 100% success across three training seeds, completing all 900 evaluation episodes. The resulting policies required an average of 42.3 primitive steps and obtained an undiscounted return of 1000 per episode.

The learned option variants discovered their first successful trajectories at the same episodes—1,155, 3,454, and 2,394—with and without replay. Their subsequent performance differed substantially. Backward replay propagated the terminal reward through the successful option sequence, allowing the learner to retain and repeatedly execute effective behavior. Safe-push options supplied structured exploration, and replay made successful experience more effective for learning. Together, these components produced reliable performance while retaining the original map, initial state, 100-step limit, and strictly 0/1000 reward scheme.''')

notebook = nbf.v4.new_notebook(cells=cells, metadata={
    'kernelspec': {'display_name': 'Python (AI6101 RL)', 'language': 'python', 'name': 'ai6101'},
    'language_info': {'name': 'python', 'version': '3.12.14'}})
path = ROOT / 'deliverables/AI6101_Code_YourName_YourMatriculationNumber.ipynb'
nbf.write(notebook, path)
print(path)
