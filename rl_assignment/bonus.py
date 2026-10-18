"""Bonus experiments: UCB exploration and sparse-reward semi-Markov Q-learning."""
from collections import deque
from pathlib import Path
import csv
import json
import time
import numpy as np
import matplotlib.pyplot as plt

from .environment import CliffBoxGridWorld
from .agent import RLAgent, evaluate_details


class ExplorationAgent(RLAgent):
    """Change only training action selection; the TD update remains Q-learning."""
    def __init__(self, env, num_episodes, strategy='ucb', c=20.0, **kwargs):
        super().__init__(env, num_episodes, **kwargs)
        if strategy not in ['epsilon_greedy', 'ucb', 'greedy'] or c < 0:
            raise ValueError('Unknown exploration strategy or negative c.')
        self.strategy, self.c = strategy, c
        self.action_counts = {state: np.zeros(4, dtype=int) for state in self.q_table}

    def act(self, state, is_training=True, rng=None):
        rng = self.rng if rng is None else rng
        if not is_training or self.strategy == 'greedy':
            return super().act(state, is_training=False, rng=rng)
        if self.strategy == 'epsilon_greedy':
            return super().act(state, is_training=True, rng=rng)
        counts = self.action_counts[state]
        untried = np.flatnonzero(counts == 0)
        if len(untried):
            return self.action_space[int(rng.choice(untried))]
        values = np.array(list(self.q_table[state].values()))
        scores = values + self.c * np.sqrt(np.log(counts.sum() + 1) / counts)
        choices = np.flatnonzero(scores == scores.max())
        return self.action_space[int(rng.choice(choices))]

    def update(self, state, intended_action, reward, next_state, done):
        self.action_counts[state][self.action_space.index(intended_action)] += 1
        super().update(state, intended_action, reward, next_state, done)


def run_exploration_bonus(output_dir='results', episodes=30000,
                          seeds=(7, 17, 27), eval_episodes=300,
                          reference_study=None):
    """Clean, original-reward comparison. Reuse the identical epsilon reference."""
    output_dir = Path(output_dir)
    methods = ['epsilon_greedy', 'ucb', 'greedy']
    history = np.zeros((len(seeds), len(methods), episodes))
    success_history = np.zeros_like(history, dtype=bool)
    evaluation = np.zeros((len(seeds), len(methods), 2))
    rows, eval_records = [], {}
    for k, seed in enumerate(seeds):
        for j, method in enumerate(methods):
            if method == 'epsilon_greedy' and reference_study is not None:
                cfg = reference_study['summary']['config']
                assert cfg['episodes'] == episodes and cfg['seeds'] == list(seeds)
                history[k,j] = reference_study['arrays']['training_returns'][k,0]
                success_history[k,j] = reference_study['arrays']['training_success'][k,0]
                evaluation[k,j] = [reference_study['arrays']['success'][k,0,0],
                                   reference_study['arrays']['mean_return'][k,0,0]]
                source = 'identical mandatory clean-training run (reused)'
            else:
                agent = ExplorationAgent(CliffBoxGridWorld(), episodes,
                                         strategy=method, c=20, seed=seed)
                history[k,j] = agent.learn()
                success_history[k,j] = agent.training_success
                details = evaluate_details(agent, 0, eval_episodes,
                                           100000 + seed * 100)
                eval_records[f'{method}_seed{seed}'] = details
                evaluation[k,j] = [np.mean(details['success']), np.mean(details['returns'])]
                source = 'new independent training run'
            indices = np.flatnonzero(success_history[k,j])
            first = int(indices[0]+1) if len(indices) else None
            rows.append(dict(seed=seed, method=method,
                             first_success=first,
                             train_success_last2000=float(success_history[k,j,-2000:].mean()),
                             eval_success=float(evaluation[k,j,0]),
                             eval_return=float(evaluation[k,j,1]), source=source))
            print(f'Exploration: seed={seed}, {method}, first success={first}, '
                  f'greedy success={evaluation[k,j,0]:.1%}', flush=True)
    arrays = dict(returns=history, success=success_history, evaluation=evaluation)
    np.savez_compressed(output_dir/'bonus_exploration.npz', **arrays)
    summary = dict(methods=methods, seeds=list(seeds), episodes=episodes,
                   eval_episodes=eval_episodes, ucb_c=20,
                   epsilon_reference_reused=reference_study is not None, rows=rows)
    (output_dir/'bonus_exploration.json').write_text(json.dumps(summary, indent=2))
    (output_dir/'bonus_exploration_evaluation.json').write_text(json.dumps(eval_records))
    with (output_dir/'bonus_exploration.csv').open('w', newline='') as file:
        writer=csv.DictWriter(file, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return dict(arrays=arrays, summary=summary)


def sparse_reward(env):
    """The only reward used by all sparse learners and sparse evaluations."""
    return 1000.0 if np.array_equal(env.box_pos, env.goal_pos) else 0.0


class SparsePrimitiveAgent(RLAgent):
    """Ordinary epsilon-greedy Q-learning with only 0/1000 rewards."""
    def learn(self, action_noise=0.0):
        if action_noise != 0:
            raise ValueError('The bonus isolates sparse learning at action noise zero.')
        rewards = []
        self.training_success, self.training_lengths = [], []
        for _ in range(self.num_episodes):
            state, done, total = self.env.reset(), False, 0.0
            while not done:
                action = self.act(state)
                nxt, _, done, _ = self.env.step(action)
                reward = sparse_reward(self.env)
                self.update(state, action, reward, nxt, done)
                self.visits[state] += 1
                state, total = nxt, total + reward
            rewards.append(total)
            self.training_success.append(total == 1000)
            self.training_lengths.append(self.env.timesteps)
        return rewards


class SafePushOptions:
    """Hand-designed skills, NOT a global box-to-goal planner.

    A skill finds a shortest safe AGENT path to the opposite side of the box,
    then performs exactly one push. BFS treats cliffs and the box as obstacles.
    Skills filter cliff/boundary pushes. They never inspect goal distance.
    The original environment executes and counts EVERY primitive action.
    """
    def __init__(self):
        self.cache = {}
        self.cliffs = set()
        for (r0, col), (r1, _) in CliffBoxGridWorld.danger_region:
            self.cliffs.update((row, col) for row in range(r0, r1+1))
        self.forces = {a: tuple(map(int, force)) for a, force in CliffBoxGridWorld.forces.items()}

    def safe(self, pos):
        return 0 <= pos[0] < 6 and 0 <= pos[1] < 14 and pos not in self.cliffs

    def path(self, start, target, box):
        if not self.safe(target) or target == box:
            return None
        queue = deque([(start, [])]); seen = {start}
        while queue:
            position, actions = queue.popleft()
            if position == target:
                return actions
            for action, (dr, dc) in self.forces.items():
                nxt = (position[0]+dr, position[1]+dc)
                if self.safe(nxt) and nxt != box and nxt not in seen:
                    seen.add(nxt);queue.append((nxt, actions+[action]))
        return None

    def available(self, state):
        if state in self.cache:
            return self.cache[state]
        agent, box = tuple(state[:2]), tuple(state[2:])
        options = {}
        for action, (dr, dc) in self.forces.items():
            dest, stand = (box[0]+dr, box[1]+dc), (box[0]-dr, box[1]-dc)
            # Match the supplied box's hard boundaries, rather than generic clipping.
            if not (1 <= dest[0] <= 4 and 1 <= dest[1] <= 13 and self.safe(dest)):
                continue
            path = self.path(agent, stand, box)
            if path is not None:
                options[action] = tuple(path+[action])
        self.cache[state] = options
        return options

    def finish_dead_end(self, env):
        """Use real safe walking steps until the original 100-step timeout."""
        done = False
        while not done:
            agent, box = tuple(map(int, env.agent_pos)), tuple(map(int, env.box_pos))
            for action, (dr, dc) in self.forces.items():
                raw = (agent[0]+dr, agent[1]+dc)
                nxt = (max(0,min(5,raw[0])), max(0,min(13,raw[1])))
                if self.safe(nxt) and nxt != box:
                    _, _, done, _ = env.step(action)
                    break
            else:
                raise RuntimeError('No safe primitive fallback in a dead-end state.')


class SparseOptionAgent:
    """Semi-Markov Q-learning over generic push skills; optional success replay."""
    def __init__(self, env, num_episodes, seed=7, replay_passes=10,
                 random_options=False, alpha=.1, gamma=.99, epsilon=.1):
        self.env, self.num_episodes = env, num_episodes
        self.rng = np.random.default_rng(seed)
        self.skills = SafePushOptions()
        self.q_table, self.option_counts = {}, {}
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.replay_passes, self.random_options = replay_passes, random_options
        self.training_success, self.training_lengths = [], []
        self.replay_updates, self.online_updates = 0, 0

    def values(self, state, create=True):
        if state not in self.q_table:
            values = {a:0.0 for a in self.skills.available(state)}
            if not create:
                return values
            self.q_table[state] = values
        return self.q_table[state]

    def select(self, state, training, rng):
        values = self.values(state, create=training)
        if not values:
            return None
        if self.random_options or (training and rng.random() < self.epsilon):
            return int(rng.choice(list(values)))
        best = max(values.values())
        return int(rng.choice([a for a,v in values.items() if v == best]))

    def update_option(self, state, option, discounted_reward, nxt, done, duration):
        target = discounted_reward
        if not done:
            target += self.gamma**duration * max(self.values(nxt).values(), default=0.0)
        current = self.values(state)[option]
        self.q_table[state][option] += self.alpha*(target-current)

    def episode(self, training=True, rng=None):
        rng = self.rng if rng is None else rng
        state, done = self.env.reset(), False
        trajectory, primitive_actions = [], []
        total, option_choices = 0.0, 0
        while not done:
            option = self.select(state, training, rng)
            if option is None:
                self.skills.finish_dead_end(self.env)
                break
            discounted, duration = 0.0, 0
            for action in self.skills.available(state)[option]:
                nxt, _, done, _ = self.env.step(action)
                reward = sparse_reward(self.env)
                discounted += self.gamma**duration * reward
                total += reward; duration += 1;primitive_actions.append(action)
                if done:
                    break
            transition = (state, option, discounted, nxt, done, duration)
            if training and not self.random_options:
                self.update_option(*transition);self.online_updates += 1
            trajectory.append(transition)
            option_choices += 1;state = nxt
        if training and total == 1000 and not self.random_options:
            # Repeat only genuine observed transitions; no demonstrations or relabeling.
            for _ in range(self.replay_passes):
                for transition in reversed(trajectory):
                    self.update_option(*transition);self.replay_updates += 1
        return dict(success=total == 1000, returns=total,
                    lengths=self.env.timesteps, options=option_choices,
                    actions=primitive_actions)

    def learn(self):
        records = []
        for _ in range(self.num_episodes):
            result=self.episode(training=True);records.append(result['returns'])
            self.training_success.append(result['success'])
            self.training_lengths.append(result['lengths'])
        return records


def evaluate_sparse(agent, total_run=300, seed=10007):
    """Freeze the learner; evaluate original starts/horizon with sparse rewards."""
    rng=np.random.default_rng(seed)
    records={'success':[], 'returns':[], 'lengths':[]}
    if isinstance(agent, SparseOptionAgent):
        # Unseen states use temporary zeros, never new Q entries during evaluation.
        for _ in range(total_run):
            result=agent.episode(training=False, rng=rng)
            for key in records: records[key].append(result[key])
    else:
        env=CliffBoxGridWorld()
        for _ in range(total_run):
            state,done,total=env.reset(),False,0.0
            while not done:
                action=agent.act(state,is_training=False,rng=rng)
                state,_,done,_=env.step(action)
                total += sparse_reward(env)
            records['success'].append(total==1000)
            records['returns'].append(total)
            records['lengths'].append(env.timesteps)
    assert np.isclose(np.mean(records['returns']),1000*np.mean(records['success']))
    return records


def run_sparse_bonus(output_dir='results', episodes=5000,
                     seeds=(7,17,27), eval_episodes=300):
    """Ablations isolate skill prior and reverse replay. Equal episode/horizon caps."""
    output_dir=Path(output_dir)
    methods=['primitive_epsilon','random_options','option_q','option_q_replay']
    returns=np.zeros((len(seeds),len(methods),episodes))
    lengths=np.zeros_like(returns,dtype=int)
    evaluation=np.zeros((len(seeds),len(methods),3))
    rows, eval_records, representative = [],{},{}
    for k,seed in enumerate(seeds):
        for j,method in enumerate(methods):
            if method=='primitive_epsilon':
                agent=SparsePrimitiveAgent(CliffBoxGridWorld(),episodes,seed=seed)
            else:
                agent=SparseOptionAgent(CliffBoxGridWorld(),episodes,seed=seed,
                     random_options=method=='random_options',
                     replay_passes=10 if method=='option_q_replay' else 0)
            returns[k,j]=agent.learn()
            lengths[k,j]=agent.training_lengths
            details=evaluate_sparse(agent,eval_episodes,200000+seed*100)
            eval_records[f'{method}_seed{seed}']=details
            evaluation[k,j]=[np.mean(details[x]) for x in ['success','returns','lengths']]
            firsts=np.flatnonzero(returns[k,j]>0)
            first=int(firsts[0]+1) if len(firsts) else None
            first_steps=int(lengths[k,j,:first].sum()) if first else None
            row=dict(seed=seed,method=method,first_success=first,
                     interactions_to_first_success=first_steps,
                     total_interactions=int(lengths[k,j].sum()),
                     train_success_last1000=float((returns[k,j,-1000:]>0).mean()),
                     eval_success=float(evaluation[k,j,0]),
                     eval_return=float(evaluation[k,j,1]),
                     eval_steps=float(evaluation[k,j,2]),
                     online_updates=getattr(agent,'online_updates',int(lengths[k,j].sum())),
                     replay_updates=getattr(agent,'replay_updates',0))
            rows.append(row)
            if k==0:
                representative[method]=agent
            print(f'Sparse: seed={seed}, {method}, first={first}, '
                  f'eval success={evaluation[k,j,0]:.1%}',flush=True)
    arrays=dict(returns=returns,lengths=lengths,evaluation=evaluation)
    np.savez_compressed(output_dir/'bonus_sparse.npz',**arrays)
    summary=dict(methods=methods,seeds=list(seeds),episodes=episodes,
                 eval_episodes=eval_episodes,replay_passes=10,rows=rows,
                 priors='Safe local BFS positioning + single safe push; no global box route, '
                        'no goal-distance features, no demonstrations, no intermediate reward.')
    (output_dir/'bonus_sparse.json').write_text(json.dumps(summary,indent=2))
    (output_dir/'bonus_sparse_evaluation.json').write_text(json.dumps(eval_records))
    with (output_dir/'bonus_sparse.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    # Replay the predeclared seed-7 learned hierarchy for the saved demonstration.
    agent=representative['option_q_replay']
    result=agent.episode(training=False,rng=np.random.default_rng(9007))
    env=CliffBoxGridWorld();state=env.reset();states=[state];rewards=[]
    for action in result['actions']:
        state,_,done,_=env.step(action);states.append(state);rewards.append(sparse_reward(env))
    rollout=dict(states=states,actions=result['actions'],rewards=rewards,success=result['success'])
    (output_dir/'bonus_sparse_rollout.json').write_text(json.dumps(rollout,indent=2))
    np.savez_compressed(output_dir/'bonus_sparse_policy_seed7.npz',
                       states=np.array(list(agent.q_table)),
                       q=np.array([[values.get(a,np.nan) for a in [1,2,3,4]]
                                   for values in agent.q_table.values()]))
    return dict(arrays=arrays,summary=summary,rollout=rollout)


def plot_bonus_results(exploration, sparse, output_dir='results'):
    output_dir=Path(output_dir)
    fig,axes=plt.subplots(1,2,figsize=(10.5,3.0),layout='constrained')
    colors=['#2463a0','#e38b2c','#37926e','#a45ca5']
    for ax,result,key,window,title in [(axes[0],exploration,'success',500,'Original reward: exploration'),
                                     (axes[1],sparse,'returns',100,'Sparse reward: skill/replay ablation')]:
        for j,method in enumerate(result['summary']['methods']):
            history=result['arrays'][key][:,j]
            if key=='returns':history=history/1000
            smooth=np.array([np.convolve(row,np.ones(window)/window,mode='valid') for row in history])
            mean,sd=smooth.mean(0),smooth.std(0,ddof=1)
            x=np.arange(window,history.shape[1]+1)
            ax.plot(x,mean,label=method,color=colors[j],lw=1.3)
            ax.fill_between(x,np.clip(mean-sd,0,1),np.clip(mean+sd,0,1),color=colors[j],alpha=.12)
        ax.set(xlabel='Training episode',ylabel='Goal-reaching rate',ylim=(-.02,1.02),title=title)
        ax.legend(fontsize=6.5,frameon=False);ax.grid(alpha=.2)
    fig.suptitle('Means across 3 seeds; bands = ±1 sample SD',fontsize=9)
    fig.savefig(output_dir/'bonus_learning_curves.png',dpi=200,bbox_inches='tight')
    return fig


if __name__ == '__main__':
    directory=Path('results');directory.mkdir(parents=True,exist_ok=True)
    reference=None
    if (directory/'summary.json').exists() and (directory/'experiment_arrays.npz').exists():
        reference={'arrays':dict(np.load(directory/'experiment_arrays.npz')),
                   'summary':json.loads((directory/'summary.json').read_text())}
    exploration=run_exploration_bonus(directory,reference_study=reference)
    sparse=run_sparse_bonus(directory)
    plot_bonus_results(exploration,sparse,directory)
