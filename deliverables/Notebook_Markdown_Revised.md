# AI6101 — Q-learning for CliffBoxPushing
**Name:** YourName · **Matriculation number:** YourMatriculationNumber

This project trains a tabular Q-learning agent to push the box to the goal. The experiments examine learning progress, the learned value function and policy, action-noise robustness, exploration strategies, and hierarchical learning with sparse rewards.

---

## 1. Grid World and State Representation
The environment contains 6 rows and 14 columns. The agent starts at `(4,1)`, the box at `(1,1)`, and the goal is `(4,13)`. Actions 1–4 correspond to up, down, left, and right.

Each state contains both positions:

$$s=(a_r,a_c,b_r,b_c).$$

The Q-table contains $84^2=7,056$ Cartesian states and four actions per state, giving 28,224 entries.

After each transition, the original reward is

$$r=-1-d_1(A,B)-d_1(B,G)-1000\mathbb{1}_{\mathrm{cliff}}+1000\mathbb{1}_{\mathrm{goal}},$$

where $d_1$ is Manhattan distance. An episode ends when the agent or box enters a cliff, the box reaches the goal, or 100 primitive steps have been executed. The box is restricted to rows 1–4 and columns 1–13; the agent can use the outer rows and columns to reposition.

---

## 2. Q-learning and Action Noise
Q-values are initialized to zero. Training uses ε-greedy action selection with ε=0.1, learning rate α=0.1, and discount factor γ=0.99. Ties between maximizing actions are resolved uniformly at random.

The update is

$$Q(s,a)\leftarrow Q(s,a)+\alpha\left[r+\gamma(1-d)\max_bQ(s',b)-Q(s,a)\right],$$

where $d$ is the terminal flag. Terminal transitions use the immediate reward as the target.

With probability $p$, the selected action is replaced uniformly by one of the other three actions. The environment executes the resulting action, and Q-learning updates the entry for the intended action. This learns the expected outcome of sending each action through the noisy transition channel.

Evaluation uses greedy action selection, fixed Q-values, and the specified test noise. Success is recorded when the box reaches the goal.

---

## 3. Experimental Setup
Agents are trained with action-noise levels $p_{\mathrm{train}}\in\{0,0.1,0.2\}$ and random seeds 7, 17, and 27. Each policy receives 30,000 training episodes with the same Q-learning parameters and original reward function.

Each trained policy is evaluated for 300 episodes at every test-noise level $p_{\mathrm{test}}\in\{0,0.1,0.2\}$. This gives 8,100 evaluation episodes, with 900 episodes contributing to each train/test-noise matrix cell. A uniform-random baseline is evaluated for 1,000 episodes.

Training curves use 500-episode moving averages. Lines show the mean across three seeds, and shaded bands show one sample standard deviation. The evaluation matrices report the mean and sample standard deviation across the three trained policies.

---

## 4. State Values and Learned Policy
The state-value estimate and greedy policy are obtained from the Q-table:

$$V(s)=\max_aQ(s,a),\qquad \pi(s)\in\arg\max_aQ(s,a).$$

Four policy slices fix the box at `(1,1)`, `(4,5)`, `(1,10)`, and `(2,13)` while varying the agent position. Colors show state values, arrows show unique greedy actions, dots show tied actions, and gray cells show unvisited states. B marks the box and G marks the goal.

The complete joint-state plots use agent and box indices defined by $14\times\mathrm{row}+\mathrm{column}$. The highest-valued visited states and their greedy actions are also listed.

---

## 5. Action-Noise Results and Greedy Trajectory
Clean-trained policies achieve 100% success without test noise. Their success rates decrease to 48.22% at test noise 0.1 and 18.56% at test noise 0.2. Action substitutions near cliffs interrupt trajectories that succeed under deterministic execution.

Agents trained with noise 0.1 and 0.2 record no goal visits across 90,000 training episodes per setting and achieve zero success across the three test-noise levels. Their noise-free greedy evaluations end in timeout. Successful goal-reaching behavior did not emerge under these training settings.

Training success includes ε exploration and action noise, while evaluation removes ε exploration. The seed-7 clean-trained greedy policy completes the task in 39 primitive steps with an undiscounted return of 577. The following trajectory and animation show its actions, states, and rewards.

---

## 6. Comparison of Exploration Strategies
The exploration study compares ε-greedy training with ε=0.1, UCB action selection, and greedy training with ε=0. All methods use the original reward, primitive actions, zero action noise, α=0.1, γ=0.99, and 30,000 training episodes per seed. The clean ε-greedy runs from the preceding experiment provide the ε-greedy results. Each policy is evaluated for 300 episodes per seed with exploration disabled.

UCB first selects actions that have not been tried in the current state. Once all actions have been sampled, it maximizes

$$Q(s,a)+20\sqrt{\frac{\ln(N(s)+1)}{N(s,a)}},$$

where $N(s,a)$ counts selections of action $a$ in state $s$, and $N(s)=\sum_aN(s,a)$. The exploration term is used for action selection; Q-values are updated using the original Q-learning target. All three strategies use random tie-breaking.

---

## 7. Sparse-Reward Learning with Push Options and Replay
The sparse experiment uses reward 1000 when the box reaches the original goal and 0 for every other transition. The map, initial positions, transition rules, and 100-primitive-step episode limit are retained.

The hierarchical learner uses hand-designed safe-push options. For each feasible push direction, local BFS finds a safe path that positions the agent on the required side of the box, followed by one push. The options use the map and box-boundary rules, while the high-level Q-table learns which push direction to select. Every primitive action is executed through the environment and counted toward the episode limit. In states with no feasible push, the agent walks safely until timeout.

For an option lasting $k$ primitive steps, the semi-Markov Q-learning target is

$$y=\sum_{i=0}^{k-1}\gamma^ir_i+(1-d)\gamma^k\max_oQ(s',o).$$

The replay variant performs 10 backward update passes over the option transitions of each successful training episode.

Four methods are compared: primitive ε-greedy Q-learning, random selection of safe-push options, option Q-learning, and option Q-learning with successful-trajectory replay. Each method trains for 5,000 episodes using seeds 7, 17, and 27, followed by 300 evaluation episodes per seed. Learned policies are evaluated greedily; the random-options baseline continues to select options uniformly. Environment interactions and replay updates are recorded for each method.

---

## 8. Discussion of Experimental Results
All three exploration strategies achieved 100% final success under noise-free evaluation. The median first-success episodes were 8,038 for ε-greedy, 5,106 for UCB, and 4,735 for greedy training. UCB accelerated goal discovery relative to ε-greedy, while greedy training discovered successful trajectories earliest. Optimistic zero initialization and random tie-breaking provided exploration even with ε=0: negative updates made previously untried actions more attractive than actions already associated with negative returns.

Under purely sparse rewards, primitive Q-learning, random safe options, and option Q-learning without replay each achieved 0% final evaluation success. Option Q-learning with successful-trajectory replay achieved 100% success across three training seeds, completing all 900 evaluation episodes. The resulting policies required an average of 42.3 primitive steps and obtained an undiscounted return of 1000 per episode.

The learned option variants discovered their first successful trajectories at the same episodes—1,155, 3,454, and 2,394—with and without replay. Their subsequent performance differed substantially. Backward replay propagated the terminal reward through the successful option sequence, allowing the learner to retain and repeatedly execute effective behavior. Safe-push options supplied structured exploration, and replay made successful experience more effective for learning. Together, these components produced reliable performance while retaining the original map, initial state, 100-step limit, and strictly 0/1000 reward scheme.
