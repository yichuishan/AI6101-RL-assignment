"""Three-seed training and all nine train/test noise combinations."""
from pathlib import Path
import csv
import json
import platform
import time
import numpy as np

from .environment import CliffBoxGridWorld
from .agent import RLAgent, evaluate_details, random_baseline, greedy_rollout


def run_study(output_dir='results', episodes=30000, eval_episodes=300,
              seeds=(7, 17, 27), noise_levels=(0.0, 0.1, 0.2)):
    """Run independent policies; evaluation never feeds back into training."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = dict(episodes=episodes, eval_episodes=eval_episodes,
                  seeds=list(seeds), noise_levels=list(noise_levels),
                  epsilon=0.1, alpha=0.1, gamma=0.99, episode_length=100,
                  python=platform.python_version(), numpy=np.__version__)
    rewards = np.zeros((len(seeds), len(noise_levels), episodes))
    train_success = np.zeros_like(rewards, dtype=bool)
    train_lengths = np.zeros_like(rewards, dtype=int)
    success = np.zeros((len(seeds), len(noise_levels), len(noise_levels)))
    mean_return, cliff_rate, timeout_rate, mean_length = [np.zeros_like(success) for _ in range(4)]
    all_evaluations, representative_agents, rows = {}, {}, []
    start = time.perf_counter()
    for k, seed in enumerate(seeds):
        for i, p_train in enumerate(noise_levels):
            agent = RLAgent(CliffBoxGridWorld(), episodes, seed=seed)
            rewards[k, i] = agent.learn(action_noise=p_train)
            train_success[k, i] = agent.training_success
            train_lengths[k, i] = agent.training_lengths
            if k == 0:
                representative_agents[p_train] = agent
            for j, p_test in enumerate(noise_levels):
                # Same per-seed/per-test stream across training conditions.
                # Different trajectories consume it differently: not paired trials.
                eval_seed = 100000 + seed * 100 + j
                details = evaluate_details(agent, p_test, eval_episodes, eval_seed)
                all_evaluations[f'seed{seed}_train{p_train}_test{p_test}'] = details
                success[k, i, j] = np.mean(details['success'])
                mean_return[k, i, j] = np.mean(details['returns'])
                cliff_rate[k, i, j] = np.mean(details['cliff'])
                timeout_rate[k, i, j] = np.mean(details['timeout'])
                mean_length[k, i, j] = np.mean(details['lengths'])
                rows.append(dict(seed=seed, p_train=p_train, p_test=p_test,
                                 episodes=eval_episodes,
                                 success_rate=success[k, i, j],
                                 mean_return=mean_return[k, i, j],
                                 cliff_rate=cliff_rate[k, i, j],
                                 timeout_rate=timeout_rate[k, i, j],
                                 mean_steps=mean_length[k, i, j]))
            print(f'seed={seed}, p_train={p_train:.1f}: '
                  f'greedy successes={success[k, i].round(3).tolist()}, '
                  f'elapsed={time.perf_counter()-start:.1f}s', flush=True)
    baseline = random_baseline(total_run=1000, seed=12345)
    rollout = greedy_rollout(representative_agents[0.0])
    arrays = dict(training_returns=rewards, training_success=train_success,
                  training_lengths=train_lengths, success=success,
                  mean_return=mean_return, cliff_rate=cliff_rate,
                  timeout_rate=timeout_rate, mean_length=mean_length)
    np.savez_compressed(output_dir / 'experiment_arrays.npz', **arrays)
    # Non-pickle policy snapshots: row-major agent index, box index, action index.
    for noise, agent in representative_agents.items():
        q = np.array([list(values.values()) for values in agent.q_table.values()])
        visits = np.array(list(agent.visits.values()))
        np.savez_compressed(output_dir / f'policy_train_{noise:.1f}_seed_{seeds[0]}.npz',
                            q=q.reshape(84, 84, 4), visits=visits.reshape(84, 84))
    summary = dict(config=config, baseline=baseline, rollout=rollout,
                   elapsed_seconds=time.perf_counter()-start,
                   success_mean=success.mean(axis=0).tolist(),
                   success_std=success.std(axis=0, ddof=1).tolist(),
                   return_mean=mean_return.mean(axis=0).tolist(),
                   return_std=mean_return.std(axis=0, ddof=1).tolist())
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    (output_dir / 'evaluation_episodes.json').write_text(json.dumps(all_evaluations))
    with (output_dir / 'evaluation_by_seed.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('Random baseline:', np.mean(baseline['success']),
          np.mean(baseline['returns']))
    print('Mean success matrix (rows=train noise, columns=test noise):\n', success.mean(0))
    print('Mean return matrix:\n', mean_return.mean(0))
    return dict(arrays=arrays, summary=summary, agents=representative_agents)


if __name__ == '__main__':
    run_study()
