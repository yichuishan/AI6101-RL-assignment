import copy
import numpy as np
import pytest

from rl_assignment.environment import CliffBoxGridWorld
from rl_assignment.bonus import (ExplorationAgent, sparse_reward, SafePushOptions,
                                SparseOptionAgent, evaluate_sparse)


def test_ucb_tries_unvisited_and_uses_count_bonus_only_during_training():
    agent=ExplorationAgent(CliffBoxGridWorld(),0,strategy='ucb',seed=7,c=20)
    s=agent.env.reset()
    agent.action_counts[s][:]=[4,0,4,4]
    assert agent.act(s)==2
    agent.action_counts[s][:]=[1,100,100,100]
    agent.q_table[s]={1:0.,2:1.,3:1.,4:1.}
    before=copy.deepcopy(agent.action_counts)
    assert agent.act(s)==1
    assert agent.act(s,is_training=False) in [2,3,4]
    assert all(np.array_equal(before[k],v) for k,v in agent.action_counts.items())


def test_pure_greedy_training_ignores_epsilon():
    agent=ExplorationAgent(CliffBoxGridWorld(),0,strategy='greedy',epsilon=1)
    s=agent.env.reset();agent.q_table[s][3]=10
    assert all(agent.act(s)==3 for _ in range(100))


def test_sparse_reward_is_exactly_zero_or_goal_1000():
    env=CliffBoxGridWorld();env.reset()
    assert sparse_reward(env)==0
    env.agent_pos=np.array([0,3])  # cliff carries no sparse penalty
    assert sparse_reward(env)==0
    env.box_pos=env.goal_pos.copy()
    assert sparse_reward(env)==1000


def test_options_execute_real_safe_path_and_exactly_one_push():
    skills=SafePushOptions();env=CliffBoxGridWorld();s=env.reset()
    for option,actions in skills.available(s).items():
        env.reset();before=env.box_pos.copy();moves=0
        for action in actions:
            old=env.box_pos.copy();nxt,_,done,_=env.step(action)
            moves+=int(not np.array_equal(old,env.box_pos))
            assert not env._check_off_cliff(env.agent_pos)
            assert not env._check_off_cliff(env.box_pos)
        assert moves==1
        assert np.array_equal(env.box_pos,before+env.forces[option])
        assert env.timesteps==len(actions)


def test_semimarkov_target_discounts_by_primitive_duration():
    agent=SparseOptionAgent(CliffBoxGridWorld(),0,alpha=1,gamma=.9)
    s=agent.env.reset();agent.values(s)
    a=next(iter(agent.q_table[s]))
    nxt=(1,1,1,2);agent.q_table[nxt]={4:10.}
    agent.update_option(s,a,0,nxt,False,3)
    assert agent.q_table[s][a]==pytest.approx(.9**3*10)
    agent.update_option(s,a,.9**2*1000,nxt,True,3)
    assert agent.q_table[s][a]==pytest.approx(810)


def test_sparse_evaluation_freezes_values_and_uses_100_step_cap():
    agent=SparseOptionAgent(CliffBoxGridWorld(),0,seed=21)
    before=copy.deepcopy(agent.q_table)
    result=evaluate_sparse(agent,5,seed=21)
    assert agent.q_table==before
    assert max(result['lengths'])<=100
    assert set(result['returns'])<={0.,1000.}
    assert np.mean(result['returns'])==1000*np.mean(result['success'])


def test_reverse_replay_does_not_invent_rewards_or_success():
    agent=SparseOptionAgent(CliffBoxGridWorld(),10,replay_passes=10,seed=101)
    rewards=agent.learn()
    assert set(rewards)<={0.,1000.}
    if not any(rewards):
        assert agent.replay_updates==0
        assert all(v==0 for q in agent.q_table.values() for v in q.values())
