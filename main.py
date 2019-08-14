"""

### NOTICE ###
You DO NOT need to upload this file

"""
import argparse
from test import test
from environment import Environment
import warnings
warnings.filterwarnings("ignore")


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_name',default=None)
    parser.add_argument('--model_path',default=None)
    parser.add_argument('--env_name', default=None, help='environment name')

    parser.add_argument('--train_pg', action='store_true', help='whether train policy gradient')
    parser.add_argument('--test_pg', action='store_true', help='whether test policy gradient')
    parser.add_argument('--pg_type', default=None, help='[pg, pg_nor, pg_ppo]')


    parser.add_argument('--train_dqn', action='store_true', help='whether train DQN')
    parser.add_argument('--test_dqn', action='store_true', help='whether test DQN')
    parser.add_argument('--dqn_type', default=None, help='[DQN, DoubleDQN, DuelDQN, DDDQN]')

    parser.add_argument('--train_a2c', action='store_true', help='whether train mario')
    parser.add_argument('--test_a2c', action='store_true', help='whether test mario')

    parser.add_argument('--video_dir', default=None, help='output video directory')
    parser.add_argument('--do_render', action='store_true', help='whether render environment')

    args = parser.parse_args()
    return args


def run(args):

    "******Policy Gradient******"
    if args.train_pg:
        env_name = args.env_name or 'LunarLander-v2'
        env = Environment(env_name, args, atari_wrapper=False)
        if args.pg_type == 'pg':
            from agent_dir.agent_pg import AgentPG
            agent = AgentPG(env, args)
        elif args.pg_type == 'pg_nor':
            from agent_dir.agent_pg import AgentPG
            agent = AgentPG(env, args, normalization=True)
        elif args.pg_type == 'pg_ppo':
            from agent_dir.agent_pg_ppo import AgentPG
            agent = AgentPG(env, args)
        agent.train()

    if args.test_pg:
        env = Environment('LunarLander-v2', args, test=True)
        from agent_dir.agent_pg import AgentPG
        agent = AgentPG(env, args)
        test(agent, env, total_episodes=30)


    "******Deep Q Learning******"
    if args.train_dqn:
        env_name = args.env_name or 'AssaultNoFrameskip-v0'
        env = Environment(env_name, args, atari_wrapper=True)
        from agent_dir.agent_dqn import AgentDQN
        agent = AgentDQN(env, args)
        agent.train()
    if args.test_dqn:
        env = Environment('AssaultNoFrameskip-v0', args, atari_wrapper=True, test=True)
        from agent_dir.agent_dqn import AgentDQN
        agent = AgentDQN(env, args)
        test(agent, env, total_episodes=100)

    if args.train_a2c:
        from agent_dir.agent_a2c import AgentA2C
        agent = AgentA2C(None, args)
        agent.train()

    if args.test_a2c:
        env = Environment('SuperMarioBros-v0', args)
        from agent_dir.agent_a2c import AgentA2C
        agent = AgentA2C(env, args)
        test(agent, env, total_episodes=10)

if __name__ == '__main__':
    args = parse()
    run(args)
