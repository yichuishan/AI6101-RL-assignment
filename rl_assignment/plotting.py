"""Publication figures: learning, robustness and state-conditioned policies."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.animation import FuncAnimation, PillowWriter

from .environment import CliffBoxGridWorld

ARROWS = {1: '↑', 2: '↓', 3: '←', 4: '→'}
COLORS = ['#2463a0', '#e38b2c', '#37926e']


def rolling_mean(values, window=500):
    values = np.asarray(values, dtype=float)
    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    return (cumulative[window:] - cumulative[:-window]) / window


def save_figure(fig, path):
    fig.savefig(path, dpi=190, bbox_inches='tight', facecolor='white')
    return fig


def plot_learning(arrays, config, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.25), layout='constrained')
    window = 500
    for i, noise in enumerate(config['noise_levels']):
        for ax, key in zip(axes, ['training_returns', 'training_success']):
            smooth = np.array([rolling_mean(row, window)
                               for row in arrays[key][:, i]])
            x = np.arange(window, config['episodes'] + 1)
            mean, sd = smooth.mean(0), smooth.std(0, ddof=1)
            ax.plot(x, mean, color=COLORS[i], label=f'train noise = {noise:.1f}', lw=1.4)
            ax.fill_between(x, mean - sd, mean + sd, color=COLORS[i], alpha=0.14)
        axes[0].set_ylabel('Undiscounted episode return')
        axes[1].set_ylabel('Goal-reaching rate')
    for ax in axes:
        ax.set_xlabel('Training episode')
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8, frameon=False)
    axes[1].set_ylim(0, 1)
    fig.suptitle('500-episode rolling mean; band = ±1 SD across 3 training seeds', fontsize=10)
    return save_figure(fig, Path(output_dir) / 'learning_curves.png')


def plot_robustness(arrays, config, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7), layout='constrained')
    levels = [f'{p:.1f}' for p in config['noise_levels']]
    for ax, key, title, cmap in zip(axes, ['success', 'mean_return'],
            ['Goal-reaching success (%)', 'Mean undiscounted episode return'],
            ['YlGnBu', 'RdYlGn']):
        means = arrays[key].mean(0)
        sd = arrays[key].std(0, ddof=1)
        scale = 100 if key == 'success' else 1
        display = means * scale
        image = ax.imshow(display, cmap=cmap,
                          **({'vmin': 0, 'vmax': 100} if key == 'success' else {}))
        for i in range(3):
            for j in range(3):
                text = (f'{display[i,j]:.1f}\n±{sd[i,j]*scale:.1f}'
                        if key == 'success' else
                        f'{display[i,j]:.0f}\n±{sd[i,j]:.0f}')
                norm = image.norm(display[i,j])
                color = 'white' if (key == 'success' and norm > .65) else '#1a2630'
                ax.text(j, i, text, ha='center', va='center', fontsize=10, color=color)
        ax.set(xticks=range(3), xticklabels=levels, yticks=range(3), yticklabels=levels,
               xlabel='Test action noise', ylabel='Training action noise', title=title)
        fig.colorbar(image, ax=ax, shrink=.8)
    return save_figure(fig, Path(output_dir) / 'robustness_heatmaps.png')


def decorate_grid(ax, box=None):
    ax.set(xlim=(-.5, 13.5), ylim=(5.5, -.5), aspect='equal',
           xticks=range(14), yticks=range(6))
    ax.set_xticks(np.arange(-.5, 14, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 6, 1), minor=True)
    ax.grid(which='minor', color='#d3d9df', lw=.5)
    ax.tick_params(which='minor', length=0)
    ax.tick_params(labelsize=7)
    for (r0, col), (r1, _) in CliffBoxGridWorld.danger_region:
        for row in range(r0, r1 + 1):
            ax.add_patch(Rectangle((col-.5, row-.5), 1, 1, color='#4c525a', zorder=3))
            ax.text(col, row, '×', color='white', ha='center', va='center', fontsize=10, zorder=4)
    gr, gc = CliffBoxGridWorld.goal_pos
    ax.text(gc, gr, 'G', color='#148143', ha='center', va='center', weight='bold', zorder=7)
    if box is not None:
        row, col = box
        ax.add_patch(Rectangle((col-.44, row-.44), .88, .88,
                               facecolor='#e8922a', edgecolor='#703e00', zorder=5))
        ax.text(col, row, 'B', ha='center', va='center', weight='bold', zorder=6)


def visualize(q_table, visits=None, box_positions=None, output_dir='results'):
    """V(s)=max_a Q(s,a); retain all greedy ties. No division by state count."""
    v_table = {state: max(q.values()) for state, q in q_table.items()}
    policy = {state: tuple(a for a, value in q.items() if value == v_table[state])
              for state, q in q_table.items()}
    print(f'Cartesian state space: {len(q_table)}; Q entries: {len(q_table)*4}')
    if visits is None:
        visits = {s: 1 for s in q_table}
    learned = [s for s in q_table if visits[s] > 0]
    print(f'Visited decision states: {len(learned)}')
    print('Top 20 VISITED state values (undivided) and greedy policies:')
    for state in sorted(learned, key=v_table.get, reverse=True)[:20]:
        print(state, f'V={v_table[state]:.2f}', 'greedy actions=', policy[state])
    box_positions = [(1, 1), (4, 5), (1, 10), (2, 13)] if box_positions is None else box_positions
    values = [v_table[s] for s in learned]
    norm = colors.Normalize(vmin=min(values), vmax=max(values))
    cmap = plt.get_cmap('YlGnBu').copy()
    cmap.set_bad('#ededed')
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 5.4), layout='constrained')
    for ax, box in zip(axes.flat, box_positions):
        matrix = np.full((6, 14), np.nan)
        for row in range(6):
            for col in range(14):
                state = (row, col, *box)
                if visits[state] > 0:
                    matrix[row, col] = v_table[state]
        image = ax.imshow(matrix, cmap=cmap, norm=norm)
        decorate_grid(ax, box)
        for row in range(6):
            for col in range(14):
                state = (row, col, *box)
                if visits[state] and (row, col) != box:
                    acts = policy[state]
                    label = ARROWS[acts[0]] if len(acts) == 1 else '·'
                    value = norm(v_table[state])
                    ax.text(col, row, label, ha='center', va='center', fontsize=10,
                            color='white' if value > .65 else '#172533', zorder=4)
        ax.set_title(f'Box fixed at {box}; agent position varies', fontsize=9)
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=.85, label='V(s) = max Q(s,a)')
    fig.suptitle('Clean-trained policy, seed 7: gray = unvisited; dot = tied actions', fontsize=10)
    save_figure(fig, Path(output_dir) / 'value_policy_slices.png')
    # Show every joint state, rather than imply there is one policy per agent cell.
    all_values = np.array([v_table[s] if visits[s] else np.nan for s in q_table]).reshape(84, 84)
    all_actions = np.array([policy[s][0] if visits[s] and len(policy[s]) == 1 else np.nan
                            for s in q_table]).reshape(84, 84)
    full, panels = plt.subplots(1, 2, figsize=(10, 4.2), layout='constrained')
    im = panels[0].imshow(all_values, cmap=cmap, aspect='auto', norm=norm)
    full.colorbar(im, ax=panels[0], label='V(s)')
    panels[0].set_title('Value over all 7,056 joint states')
    policy_cmap = colors.ListedColormap(['#2863a4', '#50a989', '#f1bd54', '#d46b60'])
    policy_cmap.set_bad('#ededed')
    im = panels[1].imshow(all_actions, cmap=policy_cmap, vmin=.5, vmax=4.5, aspect='auto')
    full.colorbar(im, ax=panels[1], ticks=[1, 2, 3, 4], label='Greedy action: up/down/left/right')
    panels[1].set_title('Unique greedy action; gray = unvisited/tie')
    for ax in panels:
        ax.set_xlabel('Box index = 14 × row + column')
        ax.set_ylabel('Agent index = 14 × row + column')
    save_figure(full, Path(output_dir) / 'all_state_value_policy.png')
    return v_table, policy, fig, full


def plot_rollout(rollout, output_dir):
    states = np.array(rollout['states'])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 2.8), layout='constrained')
    for ax, offset, label, color in zip(axes, [0, 2], ['Agent path', 'Box path'],
                                       ['#2463a0', '#d27612']):
        decorate_grid(ax)
        positions = states[:, offset:offset+2]
        for a, b in zip(positions[:-1], positions[1:]):
            if not np.array_equal(a, b):
                ax.add_patch(FancyArrowPatch((a[1], a[0]), (b[1], b[0]),
                             arrowstyle='-|>', mutation_scale=9, color=color,
                             lw=1.6, alpha=.75, zorder=5))
        ax.scatter(positions[0,1], positions[0,0], s=60, color=color, zorder=6)
        ax.set_title(label, fontsize=10)
    fig.suptitle(f'Clean greedy rollout (seed 7): {len(rollout["actions"])} steps, '
                 f'return {sum(rollout["rewards"]):.0f}, success={rollout["success"]}', fontsize=10)
    return save_figure(fig, Path(output_dir) / 'greedy_trajectory.png')


def animate_rollout(rollout, output_dir):
    fig, ax = plt.subplots(figsize=(9, 4))
    states = rollout['states']
    def frame(index):
        ax.clear()
        state = states[index]
        decorate_grid(ax, state[2:])
        ax.scatter(state[1], state[0], s=180, color='#2463a0', zorder=8)
        ax.text(state[1], state[0], 'A', color='white', ha='center', va='center', zorder=9)
        ax.set_title(f'Clean greedy policy | step {index}/{len(states)-1} | '
                     f'return {sum(rollout["rewards"][:index]):.0f}')
        return []
    animation = FuncAnimation(fig, frame, frames=len(states), interval=220)
    animation.save(Path(output_dir) / 'greedy_rollout.gif', writer=PillowWriter(fps=5), dpi=95)
    plt.close(fig)


def make_figures(study, output_dir='results', animation=True):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays, summary = study['arrays'], study['summary']
    figures = [plot_learning(arrays, summary['config'], output_dir),
               plot_robustness(arrays, summary['config'], output_dir)]
    clean = study['agents'][0.0]
    _, _, slices, all_states = visualize(clean.q_table, clean.visits, output_dir=output_dir)
    figures += [slices, all_states, plot_rollout(summary['rollout'], output_dir)]
    if animation:
        animate_rollout(summary['rollout'], output_dir)
    return figures
