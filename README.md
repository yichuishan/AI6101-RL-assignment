# AI6101-RL-assignment
Implement and evaluate one of the Reinforcement Learning (RL) algorithms  (e.g., Q-learning, SARSA, etc.) to solve the CliffBoxPushing grid-world game.

## Completed assignment

The self-contained, executed notebook and strictly two-page report are in
[`deliverables/`](deliverables/). The submission ZIP contains just these two files.
Replace `YourName` and `YourMatriculationNumber` in the document headers and filenames
before submission. The extended Chinese explanation is
[`代码逻辑与实验详解.md`](deliverables/代码逻辑与实验详解.md).

The supplied environment is unchanged. The mandatory study uses tabular Q-learning with
alpha=0.1, gamma=0.99, epsilon=0.1 and random greedy tie-breaking. Action noise
replaces an intended action uniformly with one of the other three actions; the Q
update credits the intended action. Evaluation disables epsilon exploration.

### Reproduce

Use Python 3.12 and install `requirements.txt`, or activate the prepared cloud
environment:

```bash
source /workspace/.cloud-setup/ai6101/env.sh
python -m pytest -q tests
python -m rl_assignment.experiments
```

Opening the submission notebook and running all cells also reproduces the full
study without any imports from this repository. It writes artifacts to `results/`
in its working directory. No interactive input, GPU or API credential is needed.

The study trains nine policies: 30,000 episodes per policy, noise levels 0/0.1/0.2
and seeds 7/17/27. It then evaluates each policy for 300 episodes per test noise,
8,100 evaluation episodes in total, plus 1,000 random baseline episodes. Raw records,
per-seed CSV results, policy snapshots, figures and an animated rollout are in
[`results/`](results/).

Recorded clean-trained success rates are 100%, 48.22% and 18.56% at test noise
0/0.1/0.2. The two noisy-training settings did not discover a successful trajectory
within this budget and have zero evaluation successes. Failed runs are included.
Error bands are sample standard deviations across training seeds, not confidence
intervals.

`build_notebook.py` regenerates an **unexecuted** notebook from `rl_assignment/`;
it overwrites saved outputs, so do not run it merely to read the finished result.
`build_report.py` generates the two-page PDF from current `results/` and requires
`requirements-report.txt` plus the system DejaVu Sans font. `build_explanation.py`
regenerates the extended Chinese explanation. The report-generation dependencies
are separate from the notebook runtime dependencies.

## Both bonus tasks

Notebook sections **7** and **8** implement and evaluate both optional tasks:

- **Exploration:** original-reward epsilon-greedy, UCB (c=20), and epsilon=0
  training, 30,000 episodes with seeds 7/17/27. The clean epsilon runs are reused
  from the mandatory study. Median first-success episodes are 8,038 / 5,106 /
  4,735; all reach 100% final clean evaluation success. UCB does not beat the
  greedy baseline here; optimistic initialization and random ties matter.
- **Pure sparse reward:** exactly 1000 at the original goal, otherwise zero.
  Compare primitive epsilon-greedy, random safe push options, option Q-learning,
  and option Q-learning with 10 reverse replay passes after actual successes.
  Each trains 5,000 episodes per seed. Only the replay variant achieves 100%
  final success (900/900); each other method records zero final successes.
  The options use hand-designed local BFS for safe agent positioning and one
  push; they do not plan a full box route or use goal-distance rewards.
  Every primitive step counts toward the unchanged 100-step horizon. Interaction
  counts and additional replay updates are reported rather than treated as equal.

All bonus evaluations use 300 episodes per method/seed. Learned policies disable
exploration and updates; the random-options baseline stays random by definition.
Raw bonus records, CSV tables, policy snapshot and curves are in `results/bonus_*`.

```bash
python -m rl_assignment.bonus
```

The notebook reuses completed bonus artifacts when available; set
`RECOMPUTE_BONUS=True` to retrain. With no artifacts it trains automatically, so
the submission notebook remains self-contained. Clear caches after changing code
or parameters. The updated PDF and editable Word report both contain the bonus
results and are formatted as two pages.

Regenerate documents and archives after a new experiment:

```bash
python build_report.py
python build_word_report.py
python build_explanation.py
python package_deliverables.py
```

Word generation additionally uses `python-docx` from `requirements-report.txt`.
The automated test suite now has 13 tests covering the mandatory and bonus methods.
