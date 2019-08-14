import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical




class Model(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(state_dim, n_latent_var)
        self.fc2 = nn.Linear(n_latent_var, action_dim)        
        # Memory:
        self.actions = []
        self.states = []
        self.logprobs = []
        self.state_values = []
        self.rewards = []
        
    def forward(self, state, action=None):
        # if evaluate is True then we also need to pass an action for evaluation
        # else we return a new action from distribution
        x = F.relu(self.fc1(state))
        x = self.fc2(x)
        action_probs = F.softmax(x, dim=1)
        action_distribution = Categorical(action_probs)

        if action is None:
            action = action_distribution.sample()
            self.actions.append(action)
            self.logprobs.append(action_distribution.log_prob(action))
            return action.item()
        else:
            self.logprobs.append(action_distribution.log_prob(action))
            return action_distribution.entropy().mean()

    def clearMemory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.state_values[:]
        del self.rewards[:]
        
class AgentPG:
    def __init__(self, env, args):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = env
        self.gamma = 0.99
        self.eps_clip = 0.2
        self.K_epochs = 5

        if args.test_pg:
            if args.model_path == None:
                raise Exception('give --model_path')
        else:
            if args.folder_name == None:
                raise Exception('give --folder_name')
            self.model_dir = os.path.join('./model',args.folder_name)
            if not os.path.exists(self.model_dir):
                os.mkdir(self.model_dir)

        self.policy = Model(state_dim=self.env.observation_space.shape[0], 
                            action_dim=self.env.action_space.n,
                            n_latent_var=64).to(self.device)
        self.policy_old = Model(state_dim=self.env.observation_space.shape[0], 
                            action_dim=self.env.action_space.n,
                            n_latent_var=64).to(self.device)

        if args.test_pg:
            self.load(os.path.join(self.model_dir,'model.cpt'))

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-3)
        self.num_episodes = 1500
        self.display_freq = 10
        self.n_update = 10
        self.plot = {'steps':[], 'reward':[]}


    def save(self, save_path):
        print('save model to', save_path)
        torch.save(self.policy.state_dict(), save_path)
        
    def load(self, load_path):
        print('load model from', load_path)
        self.policy.load_state_dict(torch.load(load_path))

    def init_game_setting(self):
        pass

    def make_action(self, state, test=False):
        state = torch.FloatTensor([state]).to(self.device)

        if test:
            actions = self.policy(state)
        else:
            actions = self.policy_old(state)

        return actions

        
    def update(self):   
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward in reversed(self.policy_old.rewards):
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list in tensor
        old_states = torch.tensor(self.policy_old.states).to(self.device).detach()
        old_actions = torch.tensor(self.policy_old.actions).to(self.device).detach()
        old_logprobs = torch.tensor(self.policy_old.logprobs).to(self.device).detach()
        
        # Optimize policy for K epochs:
        all_loss = 0
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            dist_entropy = self.policy(old_states, old_actions)
            
            # Finding the ratio (pi_theta / pi_theta__old):
            logprobs = self.policy.logprobs[0].to(self.device)
            ratios = torch.exp(logprobs - old_logprobs.detach())
                
            # Finding Surrogate Loss:
            advantages = rewards
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2)  - 0.01 * dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            self.policy.clearMemory()
            all_loss += loss.mean()

        self.policy_old.clearMemory()
        self.policy_old.load_state_dict(self.policy.state_dict())
        return  all_loss / self.K_epochs

    def train(self):
        # moving average of reward
        avg_reward = None 
        best_reward = 0
        for epoch in range(self.num_episodes):
            state = self.env.reset()

            self.init_game_setting()

            done = False
            rewards = []
            while(not done):
                action = self.make_action(state)
                state_n, reward, done, _ = self.env.step(action)
                
                # Saving state and reward:
                self.policy_old.states.append(state)
                self.policy_old.rewards.append(reward)
                rewards.append(reward)
        
                state = state_n

            # for logging 
            last_reward = sum(rewards)
            avg_reward = last_reward if not avg_reward else avg_reward * 0.9 + last_reward * 0.1
     
            self.plot['steps'].append(epoch)
            self.plot['reward'].append(last_reward)

            if epoch % self.display_freq == 0:
                loss = self.update()
                print('Epochs: %d/%d | Avg reward: %f | Loss: %f | Size: %d' %
                       (epoch, self.num_episodes, avg_reward, loss, len(rewards)))
            
            if avg_reward > 100 and avg_reward > best_reward: # to pass baseline, avg. reward > 50 is enough.
                best_reward = avg_reward
                print('Save model with reward {}'.format(avg_reward))
                self.save(os.path.join(self.model_dir,'model.cpt'))
                with open(os.path.join(self.model_dir,'plot.json'), 'w') as f:
                    json.dump(self.plot,f)


            