import torch
import numpy as np
from torch.distributions import Categorical
from torch.optim import RMSprop
from torch.nn.utils import clip_grad_norm_

from a2c.environment_a2c import make_vec_envs
from a2c.storage import RolloutStorage
from a2c.actor_critic import ActorCritic

from collections import deque
import os
import json
import gc


use_cuda = torch.cuda.is_available()

class AgentA2C:
    def __init__(self, env, args):

        self.use_gae = True
        self.use_standard = False
        # Hyperparameters
        self.lr = 7e-4
        self.gamma = 0.90
        self.tau = 0.95
        self.hidden_size = 512
        self.update_freq = 5
        self.n_processes = 16
        self.seed = 7122
        self.max_steps = 1e7
        self.grad_norm = 0.5
        self.clip_param = 0.2
        self.entropy_weight = 0.05

        #######################    NOTE: You need to implement
        self.recurrent = False # <- ActorCritic._forward_rnn()
        #######################    Please check a2c/actor_critic.py
        
        self.display_freq = 4000
        self.save_freq = 20000


        if args.test_a2c:
            if args.model_path == None:
                raise Exception('give --model_path')
        else:
            if args.folder_name == None:
                raise Exception('give --folder_name')
            self.model_dir = os.path.join('./model',args.folder_name)
            if not os.path.exists(self.model_dir):
                os.mkdir(self.model_dir)

        self.plot = {'reward':[]}

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        
        self.envs = env
        if self.envs == None:
            self.envs = make_vec_envs('SuperMarioBros-v0', self.seed,
                    self.n_processes)

        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        self.obs_shape = self.envs.observation_space.shape
        self.act_shape = self.envs.action_space.n

        self.rollouts = RolloutStorage(self.update_freq, self.n_processes,
                self.obs_shape, self.act_shape, self.hidden_size) 

        self.model = ActorCritic(self.obs_shape, self.act_shape,
                self.hidden_size, self.recurrent)
        
        self.ppo_epochs = 4
        self.ppo_batch_size = 5

        if args.test_a2c:
            self.load_model(args.model_path)

        self.model = self.model.to(self.device)
        self.optimizer = RMSprop(self.model.parameters(), lr=self.lr, eps=1e-5)

        self.hidden = None
        self.init_game_setting()

    def ppo_iter(self, mini_batch_size, states, hiddens, masks, actions, log_probs, returns, advantage):
        batch_size = states.size(0)
        for _ in range(batch_size // mini_batch_size):
            rand_ids = np.random.randint(0, batch_size, mini_batch_size)
            yield states[rand_ids, :], hiddens[rand_ids, :], masks[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]

    def _update(self):

        # R_t = reward_t + gamma * R_{t+1}
        with torch.no_grad():
            Return = self.model.get_estimate_returns(self.rollouts.obs[-1],
                                                     self.rollouts.hiddens[-1],
                                                     self.rollouts.masks[-1])

        self.rollouts.value_preds[-1].copy_(Return)
        self.rollouts.returns[-1].copy_(Return * self.rollouts.masks[-1])

        if self.use_standard:
            self.rollouts.rewards = (self.rollouts.rewards - self.rollouts.rewards.mean())/self.rollouts.rewards.std()

        if self.use_gae:
            gae = 0
            for r in reversed(range(len(self.rollouts.rewards))):
                delta = self.rollouts.rewards[r] \
                        + self.gamma * self.rollouts.value_preds[r+1] * self.rollouts.masks[r+1] \
                        - self.rollouts.value_preds[r]
                gae = delta + self.gamma * self.tau * self.rollouts.masks[r+1] * gae
                Return = gae + self.rollouts.value_preds[r]
                self.rollouts.returns[r].copy_(Return)
        else:
            for r in reversed(range(len(self.rollouts.rewards))):
                Return = self.rollouts.rewards[r] + self.gamma * Return * self.rollouts.masks[r+1]
                self.rollouts.returns[r].copy_(Return)

        # Compute actor critic loss (value_loss, action_loss)
        # OPTIONAL: You can also maxmize entropy to encourage exploration
        # loss = value_loss + action_loss (- entropy_weight * entropy)
        
        #action_probs = self.rollouts.action_probs.view(self.n_processes * self.update_freq, -1)
        #est_returns = self.rollouts.value_preds[:-1].view(self.n_processes * self.update_freq, -1)
        
        with torch.no_grad():
            est_returns, log_probs, _ = self.model(
            self.rollouts.obs[:-1].view(self.n_processes * self.update_freq, *self.obs_shape),
            self.rollouts.hiddens[:-1].view(self.n_processes * self.update_freq, -1),
            self.rollouts.masks[:-1].view(self.n_processes * self.update_freq, -1),
            )
        states = self.rollouts.obs[:-1]
        hiddens = self.rollouts.hiddens[:-1]
        masks = self.rollouts.masks[:-1]
        actions = self.rollouts.actions
        returns = self.rollouts.returns[:-1]
        est_returns = est_returns.view(self.update_freq, self.n_processes, -1)
        log_probs = log_probs.gather(1,actions.view(self.n_processes * self.ppo_batch_size, -1)).view(self.update_freq, self.n_processes, -1)
        advantages = returns - est_returns

        all_loss = []

        for _ in range(self.ppo_epochs):
            for state, hidden, mask, action, old_log_probs, return_, advantage in self.ppo_iter(self.ppo_batch_size, states, hiddens, masks, actions, log_probs, returns, advantages):

                action = action.view(self.n_processes * self.ppo_batch_size, -1)
                return_ = return_.view(self.n_processes * self.ppo_batch_size, -1)
                state = state.view(self.n_processes * self.ppo_batch_size, *self.obs_shape)
                hidden = hidden.view(self.n_processes * self.ppo_batch_size, -1)
                mask = mask.view(self.n_processes * self.ppo_batch_size, -1)
                old_log_probs = old_log_probs.view(self.n_processes * self.ppo_batch_size, -1)
                advantage = advantage.view(self.n_processes * self.ppo_batch_size, -1)

                value, new_log_probs, _ = self.model(state, hidden, mask)

                ratio = (new_log_probs.gather(1,action).log() - old_log_probs.log()).exp()

                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantage

                # action loss (Policy)
                action_loss  = - torch.min(surr1, surr2).mean()
                # value loss (DQN)
                value_loss = (return_ - value).pow(2).mean()
                # entropy
                entropy = (new_log_probs * new_log_probs.log()).sum(1).mean()
                # loss
                loss = 0.5 * value_loss + action_loss - self.entropy_weight * entropy

                # Update
                self.optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.model.parameters(), self.grad_norm)
                self.optimizer.step()
                all_loss.append(loss.item())

        # Clear rollouts after update (RolloutStorage.reset())
        self.rollouts.reset()
        return sum(all_loss)/len(all_loss)

    def _step(self, obs, hiddens, masks):
        
        with torch.no_grad():
            values, action_probs, hiddens = self.model(obs, hiddens, masks)

        actions = Categorical(action_probs.detach()).sample()

        # Sample actions from the output distributions
        obs, rewards, dones, infos = self.envs.step(actions.cpu().numpy())
        obs = torch.from_numpy(obs)
        rewards = torch.from_numpy(rewards).unsqueeze(1)
        masks = torch.from_numpy(1 - (dones)).unsqueeze(1)
        actions = actions.unsqueeze(1)
    
        self.rollouts.insert(obs, #next
                             hiddens, #next
                             actions, #now
                             action_probs, #now
                             values, #now
                             rewards, #now
                             masks) #next

        # Store transitions (obs, hiddens, actions, values, rewards, masks)

        
    def train(self):

        print('Start training')
        running_reward = deque(maxlen=10)
        episode_rewards = torch.zeros(self.n_processes, 1).to(self.device)
        total_steps = 0
        best_reward = 0
        
        # Store first observation
        obs = torch.from_numpy(self.envs.reset()).to(self.device)
        self.rollouts.obs[0].copy_(obs)
        self.rollouts.to(self.device)
        
        while True:
            # Update once every n-steps
            for step in range(self.update_freq):
                self._step(
                    self.rollouts.obs[step],
                    self.rollouts.hiddens[step],
                    self.rollouts.masks[step])

                # Calculate episode rewards
                episode_rewards += self.rollouts.rewards[step]
                for r, m in zip(episode_rewards, self.rollouts.masks[step + 1]):
                    if m == 0:
                        running_reward.append(r.item())
                episode_rewards *= self.rollouts.masks[step + 1]

            loss = self._update()
            total_steps += self.update_freq * self.n_processes
           
            # Log & save model
            if len(running_reward) == 0:
                avg_reward = 0
            else:
                avg_reward = sum(running_reward) / len(running_reward)

            self.plot['reward'].append(avg_reward)

            print('Steps: %d/%d | Avg reward: %f | Loss: %f'%
                (total_steps, self.max_steps, avg_reward, loss), end='\r')

            if total_steps % self.display_freq == 0:
                print('Steps: %d/%d | Avg reward: %f'%
                        (total_steps, self.max_steps, avg_reward))  
                if total_steps % self.save_freq == 0:

                    with open(os.path.join(self.model_dir,'plot.json'), 'w') as f:
                        json.dump(self.plot,f)
                    #if int(avg_reward) > best_reward:
                    best_reward = int(avg_reward)
                    self.save_model(os.path.join(self.model_dir,'s{}_r{}_model.pt'.format(total_steps,best_reward)))

            if total_steps >= self.max_steps:
                break

    def save_model(self, path):
        torch.save({'model':self.model,'optimizer':self.optimizer.state_dict()}, path)

    def load_model(self, path):
        print('Load model from', path)
        self.model = torch.load(path)['model']

    def init_game_setting(self):
        if self.recurrent:
            self.hidden = torch.zeros(1, self.hidden_size).to(self.device)

    def make_action(self, observation, test=False):

        obs = torch.FloatTensor([observation]).to(self.device)
        #self.rollouts.obs[0].copy_(obs)
        #self.rollouts.to(self.device)
        with torch.no_grad():
            action_probs, _ = self.model.get_action_probs(                    
                    obs,
                    None,
                    None)
        action = action_probs.max(1)[1].item()
        return action
