import os
import numpy as np
import json
import torch
from torch.autograd import grad
import torch.nn as nn
import torch.nn.functional as F

from agent_dir.agent import Agent
from environment import Environment

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_num, hidden_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_num)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        action_prob = F.softmax(x, dim=1)
        return action_prob

class AgentPG(Agent):
    def __init__(self, env, args, normalization=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = env
        
        if args.test_pg:
            if args.model_path == None:
                raise Exception('give --model_path')
        else:
            if args.folder_name == None:
                raise Exception('give --folder_name')
            self.model_dir = os.path.join('./model',args.folder_name)
            if not os.path.exists(self.model_dir):
                os.mkdir(self.model_dir)

        self.model = PolicyNet(state_dim = self.env.observation_space.shape[0],
                               action_num= self.env.action_space.n,
                               hidden_dim=64).to(self.device)

        if args.test_pg:
            self.load(args.model_path)

        # discounted reward
        self.gamma = 0.99 
        self.normalization = normalization
        
        # training hyperparameters
        self.num_episodes = 1500 # total training episodes (actually too large...)
        self.display_freq = 10 # frequency to display training progress
        
        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-3)
        
        # saved rewards and actions
        self.rewards, self.saved_actions = [], []
        self.plot = {'steps':[], 'reward':[]}
    
    
    def save(self, save_path):
        print('save model to', save_path)
        torch.save(self.model.state_dict(), save_path)
        
    def load(self, load_path):
        print('Load model from', load_path)
        self.model.load_state_dict(torch.load(load_path))

    def init_game_setting(self):
        self.rewards, self.saved_actions = [], []

    def make_action(self, state, test=False):
        # Use your model to output distribution over actions and sample from it.
        state = torch.FloatTensor([state]).to(self.device)
        action_probs = self.model(state)
        c = torch.distributions.Categorical(action_probs)
        action = c.sample()

        if test:
            return action.item()
        else:
            return action.item(), c.log_prob(action)

    def update(self):
        # discount your saved reward
        rewards = []
        running_add = 0
        for r in self.rewards[::-1]:
            if r == 0:
                running_add = 0
            running_add = running_add * self.gamma + r
            rewards.insert(0,running_add)

        rewards = torch.FloatTensor(rewards).to(self.device)
        # rewards normalization
        if self.normalization: 
        	rewards = (rewards - rewards.mean()) / rewards.std()

        action_probs = torch.stack(self.saved_actions).squeeze().to(self.device)

        # compute loss
        loss = torch.sum(torch.mul(action_probs, rewards).mul(-1), -1)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def train(self):
    	# moving average of reward
        avg_reward = None 
        best_reward = 0
        for epoch in range(self.num_episodes):
            state = self.env.reset()
            self.init_game_setting()
            done = False
            while(not done):
                action, prob = self.make_action(state)
                state, reward, done, _ = self.env.step(action)
                self.rewards.append(reward)
                self.saved_actions.append(prob)

            # for logging 
            last_reward = np.sum(self.rewards)
            avg_reward = last_reward if not avg_reward else avg_reward * 0.9 + last_reward * 0.1
            # update model
            loss = self.update()
            self.plot['steps'].append(epoch)
            self.plot['reward'].append(last_reward)
            if epoch % self.display_freq == 0:
                print('Epochs: %d/%d | Avg reward: %f | Loss: %f | Size: %d' %
                       (epoch, self.num_episodes, avg_reward, loss, len(self.saved_actions)))
            
            if avg_reward > 100 and avg_reward > best_reward: # to pass baseline, avg. reward > 50 is enough.
                best_reward = avg_reward
                print('Save model with reward {}'.format(avg_reward))
                self.save(os.path.join(self.model_dir,'model.cpt'))
                with open(os.path.join(self.model_dir,'plot.json'), 'w') as f:
                    json.dump(self.plot,f)
                    
                

