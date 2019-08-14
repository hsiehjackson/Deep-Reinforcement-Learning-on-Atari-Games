"""

### NOTICE ###
You DO NOT need to upload this file

"""

import argparse
import numpy as np
from environment import Environment
from tqdm import tqdm

seed = 1000000

def test(agent, env, total_episodes=30):
    rewards = []
    env.seed(seed)
    for i in range(total_episodes):
        state = env.reset()
        agent.init_game_setting()
        done = False
        episode_reward = 0.0

        #playing one game
        while(not done):
            action = agent.make_action(state, test=True)
            state, reward, done, info = env.step(action)
            episode_reward += reward
        print('Episode:{}/{} | reward: {:.2f}'.format(i+1,total_episodes,episode_reward),end='\r')
        rewards.append(episode_reward)
    print('\nRun %d episodes'%(total_episodes))
    print('Mean:', np.mean(rewards))

def run(args):
    if args.test_pg:
        env = Environment('LunarLander-v2', args, test=True)
        from agent_dir.agent_pg import AgentPG
        agent = AgentPG(env, args)
        test(agent, env)

    if args.test_dqn:
        env = Environment('AssaultNoFrameskip-v0', args, atari_wrapper=True, test=True)
        from agent_dir.agent_dqn import AgentDQN
        agent = AgentDQN(env, args)
        test(agent, env, total_episodes=100)

    if args.test_mario:
        env = Environment('SuperMarioBros-v0', args, test=True)
        from agent_dir.agent_mario import AgentMario
        agent = AgentMario(env, args)
        test(agent, env, total_episodes=10)

if __name__ == '__main__':
    args = parse()
    run(args)
