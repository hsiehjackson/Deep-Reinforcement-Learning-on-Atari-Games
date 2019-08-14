import random
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import json
import os

from agent_dir.agent import Agent
from environment import Environment

from collections import deque, namedtuple

use_cuda = torch.cuda.is_available()

#DDQN
#Dual-DQN
#Prioritized reply
#Multi-step (MC and TD)
#Noisy Net
#Distributional DQN

class DQN(nn.Module):
    '''
    This architecture is the one from OpenAI Baseline, with small modification.
    '''
    def __init__(self, channels, num_actions, duel_net=False):
        super(DQN, self).__init__()
        self.duel_net = duel_net
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(7*7*64, 512)
        self.head = nn.Linear(512, num_actions)        
        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU(0.01)

        if self.duel_net:
            self.fc_value = nn.Linear(512, 1)
            self.fc_advantage = nn.Linear(512, num_actions)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.lrelu(self.fc(x.view(x.size(0), -1)))
        if self.duel_net:
            value = self.fc_value(x)
            advantage = self.fc_advantage(x)
            q = value + advantage - advantage.mean()
        else:
            q = self.head(x)

        return q

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.position = 0
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        if len(self.memory) < self.buffer_size:
            self.memory.append(None)
        self.memory[self.position] = e
        self.position = (self.position + 1) % self.buffer_size
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.cat([e.state for e in experiences if e is not None]).float().cuda()
        next_states = torch.cat([e.next_state for e in experiences if e is not None]).float().cuda()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().cuda()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().cuda()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().cuda()
        return (states, actions, rewards, next_states, dones)
    def __len__(self):
        return len(self.memory)


class AgentDQN(Agent):
    def __init__(self, env, args):
        self.env = env
        self.input_channels = 4
        self.num_actions = self.env.action_space.n

        if args.test_dqn:
            if args.model_path == None:
                raise Exception('give --model_path')
        else:
            if args.folder_name == None:
                raise Exception('give --folder_name')
            self.model_dir = os.path.join('./model',args.folder_name)
            if not os.path.exists(self.model_dir):
                os.mkdir(self.model_dir)

        # build target, online network
        self.dqn_type = args.dqn_type
        print('Using {} Network'.format(args.dqn_type))
        if args.dqn_type == None:
            raise Exception('give --dqn_type')
        elif args.dqn_type == 'DQN' or args.dqn_type == 'DoubleDQN':
            self.online_net = DQN(self.input_channels, self.num_actions)
            self.online_net = self.online_net.cuda() if use_cuda else self.online_net
            self.target_net = DQN(self.input_channels, self.num_actions)
            self.target_net = self.target_net.cuda() if use_cuda else self.target_net
            self.target_net.load_state_dict(self.online_net.state_dict())
        elif args.dqn_type == 'DuelDQN' or args.dqn_type == 'DDDQN':
            self.online_net = DQN(self.input_channels, self.num_actions,duel_net=True)
            self.online_net = self.online_net.cuda() if use_cuda else self.online_net
            self.target_net = DQN(self.input_channels, self.num_actions,duel_net=True)
            self.target_net = self.target_net.cuda() if use_cuda else self.target_net
            self.target_net.load_state_dict(self.online_net.state_dict())
        else:
            raise Exception('--dqn_type must in [DQN, DoubleDQN, DuelDQN, DDDQN]')

        if args.test_dqn:
            self.load(args.model_path)
        
        # discounted reward
        self.GAMMA = 0.99
        
        # training hyperparameters
        self.train_freq = 4 # frequency to train the online network
        self.num_timesteps = 3000000 # total training steps
        self.learning_start = 10000 # before we start to update our network, we wait a few steps first to fill the replay.
        self.batch_size = 32
        self.display_freq = 100 # frequency to display training progress
        self.target_update_freq = 1000 # frequency to update target network

        # optimizer
        self.optimizer = optim.RMSprop(self.online_net.parameters(), lr=1e-4)
        self.steps = 0 # num. of passed steps. this may be useful in controlling exploration
        self.eps_min = 0.025
        self.eps_max = 1.0
        self.eps_step = 200000
        self.plot = {'steps':[], 'reward':[]}

        # TODO:
        # Initialize your replay buffer
        self.memory = ReplayBuffer(10000, self.batch_size)

    def save(self, save_path):
        print('save model to', save_path)
        model = {'online': self.online_net.state_dict(), 'target': self.self.target_net.state_dict()}
        torch.save(model, save_path)

    def load(self, load_path):
        print('Load model from', load_path)
        if use_cuda:
            self.online_net.load_state_dict(torch.load(load_path)['online'])
        else:
            self.online_net.load_state_dict(torch.load(load_path, map_location=lambda storage, loc: storage)['online'])

    def init_game_setting(self):
        # we don't need init_game_setting in DQN
        pass
    
    def epsilon(self, step):
        if step > self.eps_step:
            return 0
        else:
            return self.eps_min + (self.eps_max - self.eps_min) * ((self.eps_step - step) / self.eps_step)

    def make_action(self, state, test=False):
        if test:
            state = torch.from_numpy(state).permute(2,0,1).unsqueeze(0)
            state = state.cuda() if use_cuda else state
            with torch.no_grad():
                action = self.online_net(state).max(1)[1].item()
        else:            
            if random.random() > self.epsilon(self.steps):  
                with torch.no_grad():
                    action = self.online_net(state).max(1)[1].item()
            else:
                action = random.randrange(self.num_actions)

        return action

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        experiences = self.memory.sample()
        batch_state, batch_action, batch_reward, batch_next, batch_done, = experiences

        if self.dqn_type=='DoubleDQN' or self.dqn_type == 'DDDQN':
            next_q_actions = self.online_net(batch_next).detach().max(1)[1].unsqueeze(1)
            next_q_values = self.target_net(batch_next).gather(1,next_q_actions)
        else:
            next_q_values =  self.target_net(batch_next).detach()
            next_q_values = next_q_values.max(1)[0].unsqueeze(1)
        
        batch_reward = batch_reward.clamp(-1.1)
        current_q = self.online_net(batch_state).gather(1, batch_action)            
        next_q = batch_reward + (1 - batch_done) * self.GAMMA * next_q_values
        loss = F.mse_loss(current_q, next_q)
        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()
        return loss.item()

    def train(self):
        best_reward = 0
        episodes_done_num = 1 # passed episodes
        total_reward = [] # compute average reward
        total_loss = []
        while(True):
            state = self.env.reset()
            # State: (80,80,4) --> (1,4,80,80)
            state = torch.from_numpy(state).permute(2,0,1).unsqueeze(0)
            state = state.cuda() if use_cuda else state
            done = False
            episodes_reward = []
            episodes_loss = []
            while(not done):
                # select and perform action
                action = self.make_action(state)
                next_state, reward, done, _ = self.env.step(action)
                episodes_reward.append(reward)
                total_reward.append(reward)
                # process new state
                next_state = torch.from_numpy(next_state).permute(2,0,1).unsqueeze(0)
                next_state = next_state.cuda() if use_cuda else next_state
                # TODO:
                # store the transition in memory
                self.memory.add(state, action, reward, next_state, done)
                # move to the next state
                state = next_state
                # Perform one step of the optimization
                if self.steps > self.learning_start and self.steps % self.train_freq == 0:
                    loss = self.update()
                    episodes_loss.append(loss)
                    total_loss.append(loss)
                # update target network
                if self.steps > self.learning_start and self.steps % self.target_update_freq == 0:
                    self.target_net.load_state_dict(self.online_net.state_dict())
                # save the model
                self.steps += 1
            
            avg_ep_loss = sum(episodes_loss)/len(episodes_loss) if len(episodes_loss) > 0 else 0

            print('Episode: %d | Steps: %d/%d | Avg reward: %f | Loss: %f'% 
                 (episodes_done_num, self.steps, self.num_timesteps, 
                  sum(episodes_reward), avg_ep_loss),end='\r')

            self.plot['steps'].append(episodes_done_num)
            self.plot['reward'].append(sum(episodes_reward))

            if episodes_done_num % self.display_freq == 0:

                avg_reward = sum(total_reward) / self.display_freq
                avg_loss = sum(total_loss) / len(total_loss) if len(total_loss) > 0 else 0

                if self.steps < self.eps_step:
                    phase = "Exploring phase"
                else:
                    phase = "Learning phase"

                print('%s | Episode: %d | Steps: %d/%d | epsilon: %f | Avg reward: %f | Loss: %f'% 
                        (phase, episodes_done_num, self.steps, self.num_timesteps,
                            self.epsilon(self.steps), avg_reward, avg_loss))

                if avg_reward > best_reward and self.steps > self.eps_step:
                    best_reward = avg_reward
                    self.save(os.path.join(self.model_dir, 'e{}_r{:.2f}_model.cpt'.format(episodes_done_num, avg_reward)))
                    with open(os.path.join(self.model_dir, 'plot.json'), 'w') as f:
                        json.dump(self.plot,f)

                total_reward = []
                total_loss = []

            episodes_done_num += 1

            if self.steps > self.num_timesteps:
                break

        self.save(os.path.join(self.model_dir,'e{}_model.cpt'.format(episodes_done_num)))
        with open(os.path.join(self.model_dir, 'plot.json'), 'w') as f:
            json.dump(self.plot,f)
