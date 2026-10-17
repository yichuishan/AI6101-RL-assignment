import copy
import numpy as np
import pytest

from rl_assignment.agent import RLAgent, apply_action_noise, evaluate_details
from rl_assignment.environment import CliffBoxGridWorld


def test_q_learning_bootstrap_and_terminal_mask():
    agent = RLAgent(CliffBoxGridWorld(), 0, alpha=0.5, gamma=0.9)
    s, nxt = (4, 1, 1, 1), (3, 1, 1, 1)
    agent.q_table[nxt][1] = 10
    agent.update(s, 2, -2, nxt, False)
    assert agent.q_table[s][2] == pytest.approx(3.5)
    # A terminal next state with large Q must not contaminate the target.
    agent.update(s, 3, -2, nxt, True)
    assert agent.q_table[s][3] == pytest.approx(-1)


def test_noise_replaces_only_with_other_actions_and_is_uniform():
    rng = np.random.default_rng(21)
    draws = [apply_action_noise(1, [1, 2, 3, 4], 1, rng) for _ in range(30000)]
    counts = np.bincount(draws, minlength=5)
    assert counts[1] == 0
    assert np.all(np.abs(counts[2:5] / 30000 - 1 / 3) < 0.015)
    assert all(apply_action_noise(1, [1, 2, 3, 4], 0, rng) == 1
               for _ in range(100))
    draws = [apply_action_noise(1, [1, 2, 3, 4], 0.2, rng) for _ in range(30000)]
    assert np.mean(np.array(draws) != 1) == pytest.approx(0.2, abs=0.01)


def test_eval_disables_exploration_and_does_not_update_q():
    agent = RLAgent(CliffBoxGridWorld(), 0, epsilon=1, seed=1)
    for values in agent.q_table.values():
        values[1] = 2
    before = copy.deepcopy(agent.q_table)
    records = evaluate_details(agent, 0, total_run=2)
    # Always up: agent reaches row 0, then times out. Epsilon=1 must be ignored.
    assert records['lengths'] == [100, 100]
    assert records['success'] == [False, False]
    assert agent.q_table == before


class OneStepEnvironment:
    action_space = [1, 2, 3, 4]
    world_height, world_width = 1, 2
    goal_pos = np.array([0, 1])
    def reset(self):
        self.agent_pos = np.array([0, 0])
        self.box_pos = np.array([0, 0])
        self.timesteps = 0
        return (0, 0, 0, 0)
    def step(self, action):
        self.executed = action
        self.timesteps = 1
        return (0, 0, 0, 0), 10, True, {}


def test_training_credits_intended_not_executed_action():
    env = OneStepEnvironment()
    agent = RLAgent(env, 1, epsilon=0, alpha=1)
    state = (0, 0, 0, 0)
    agent.q_table[state][1] = 1
    agent.learn(action_noise=1)
    assert env.executed != 1
    assert agent.q_table[state][1] == 10
    assert agent.q_table[state][env.executed] == 0


def test_original_environment_timeout_push_and_cliff():
    env = CliffBoxGridWorld(episode_length=2)
    assert env.reset() == (4, 1, 1, 1)
    _, _, done, _ = env.step(1)
    assert not done
    _, _, done, _ = env.step(1)
    assert done
    env.reset()
    env.agent_pos, env.box_pos = np.array([2, 1]), np.array([1, 1])
    env.step(1)
    assert np.array_equal(env.box_pos, [1, 1])  # box cannot enter row 0
    assert np.array_equal(env.agent_pos, [2, 1])
    env.reset()
    env.agent_pos = np.array([4, 3])
    _, reward, done, _ = env.step(1)  # (3, 3) is a cliff
    assert done and reward < -1000


def test_success_metric_uses_goal_not_positive_return():
    class GoalWithNegativeReturn(OneStepEnvironment):
        def step(self, action):
            self.box_pos = self.goal_pos.copy()
            self.timesteps = 1
            return (0, 0, 0, 1), -20, True, {}
        def _check_off_cliff(self, pos):
            return False
    agent = RLAgent(OneStepEnvironment(), 0)
    records = evaluate_details(agent, 0, total_run=1,
                               env_factory=GoalWithNegativeReturn)
    assert records['success'] == [True]
    assert records['returns'] == [-20]
