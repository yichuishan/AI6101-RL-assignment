"""Tabular Q-learning and independent, noise-aware policy evaluation."""
import numpy as np


def apply_action_noise(action, action_space, probability, rng):
    """Replace action uniformly by ONE OF THE OTHER actions with probability p."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError('Action-noise probability must be in [0, 1].')
    if probability > 0 and rng.random() < probability:
        alternatives = [a for a in action_space if a != action]
        return int(rng.choice(alternatives))
    return int(action)


class RLAgent:
    def __init__(self, env, num_episodes, epsilon=0.1, alpha=0.1,
                 gamma=0.99, seed=7):
        if not 0 <= epsilon <= 1 or not 0 < alpha <= 1 or not 0 <= gamma < 1:
            raise ValueError('Invalid epsilon, alpha or gamma.')
        self.action_space = list(env.action_space)
        self.q_table = {
            (ar, ac, br, bc): {a: 0.0 for a in self.action_space}
            for ar in range(env.world_height)
            for ac in range(env.world_width)
            for br in range(env.world_height)
            for bc in range(env.world_width)
        }
        self.visits = {state: 0 for state in self.q_table}
        self.env = env
        self.num_episodes = num_episodes
        self.epsilon, self.alpha, self.gamma = epsilon, alpha, gamma
        self.rng = np.random.default_rng(seed)
        self.training_success, self.training_lengths = [], []

    def act(self, state, is_training=True, rng=None):
        """Epsilon-greedy during training; greedy with random ties otherwise."""
        rng = self.rng if rng is None else rng
        if is_training and rng.random() < self.epsilon:
            return int(rng.choice(self.action_space))
        values = self.q_table[state]
        best = max(values.values())
        choices = [a for a, value in values.items() if value == best]
        # No unnecessary random draw when the greedy action is unique.
        return choices[0] if len(choices) == 1 else int(rng.choice(choices))

    def update(self, state, intended_action, reward, next_state, done):
        """Terminal transitions have no bootstrap, including the 100-step cap."""
        target = float(reward)
        if not done:
            target += self.gamma * max(self.q_table[next_state].values())
        current = self.q_table[state][intended_action]
        self.q_table[state][intended_action] += self.alpha * (target - current)

    def learn(self, action_noise=0.0):
        """Learn Q(s, intended action) for the noisy action channel."""
        if not 0 <= action_noise <= 1:
            raise ValueError('Action-noise probability must be in [0, 1].')
        rewards = []
        self.training_success, self.training_lengths = [], []
        for _ in range(self.num_episodes):
            state = self.env.reset()
            cumulative_reward = 0.0
            done = False
            while not done:
                intended = self.act(state)
                executed = apply_action_noise(intended, self.action_space,
                                              action_noise, self.rng)
                next_state, reward, done, _ = self.env.step(executed)
                # Credit the decision we controlled, not the corrupted action.
                self.update(state, intended, reward, next_state, done)
                self.visits[state] += 1
                state = next_state
                cumulative_reward += reward
            rewards.append(cumulative_reward)
            self.training_success.append(bool(np.array_equal(
                self.env.box_pos, self.env.goal_pos)))
            self.training_lengths.append(self.env.timesteps)
        return rewards


def evaluate(agent, p_test, total_run=300, seed=10007, env_factory=None):
    """No learning or epsilon exploration; returns success rate and mean return."""
    details = evaluate_details(agent, p_test, total_run, seed, env_factory)
    return float(np.mean(details['success'])), float(np.mean(details['returns']))


def evaluate_details(agent, p_test, total_run=300, seed=10007, env_factory=None):
    if total_run < 1 or not 0 <= p_test <= 1:
        raise ValueError('Need positive episodes and noise in [0, 1].')
    env = (CliffBoxGridWorld if env_factory is None else env_factory)()
    rng = np.random.default_rng(seed)
    records = {'returns': [], 'success': [], 'lengths': [], 'cliff': [], 'timeout': []}
    for _ in range(total_run):
        state, done, episode_return = env.reset(), False, 0.0
        while not done:
            intended = agent.act(state, is_training=False, rng=rng)
            executed = apply_action_noise(intended, agent.action_space, p_test, rng)
            state, reward, done, _ = env.step(executed)
            episode_return += reward
        success = bool(np.array_equal(env.box_pos, env.goal_pos))
        cliff = bool(env._check_off_cliff(env.agent_pos) or
                     env._check_off_cliff(env.box_pos))
        records['returns'].append(float(episode_return))
        records['success'].append(success)
        records['lengths'].append(env.timesteps)
        records['cliff'].append(cliff)
        records['timeout'].append(not success and not cliff)
    return records


def random_baseline(total_run=1000, seed=12345, env_factory=None):
    """Uniform random actions, using the same environment and success definition."""
    env = (CliffBoxGridWorld if env_factory is None else env_factory)()
    rng = np.random.default_rng(seed)
    records = {'returns': [], 'success': [], 'lengths': []}
    for _ in range(total_run):
        state, done, episode_return = env.reset(), False, 0.0
        while not done:
            state, reward, done, _ = env.step(int(rng.choice(env.action_space)))
            episode_return += reward
        records['returns'].append(float(episode_return))
        records['success'].append(bool(np.array_equal(env.box_pos, env.goal_pos)))
        records['lengths'].append(env.timesteps)
    return records


def greedy_rollout(agent, seed=9007, env_factory=None):
    """Record one clean greedy trajectory for inspection and visualization."""
    env = (CliffBoxGridWorld if env_factory is None else env_factory)()
    rng = np.random.default_rng(seed)
    state, done = env.reset(), False
    states, actions, rewards = [state], [], []
    while not done:
        action = agent.act(state, is_training=False, rng=rng)
        state, reward, done, _ = env.step(action)
        states.append(state)
        actions.append(action)
        rewards.append(float(reward))
    return {'states': states, 'actions': actions, 'rewards': rewards,
            'success': bool(np.array_equal(env.box_pos, env.goal_pos))}


# Import at the bottom so this file's definitions can also form a standalone
# Notebook cell immediately after the original environment cell.
from .environment import CliffBoxGridWorld
