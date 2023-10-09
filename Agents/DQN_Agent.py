import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
from Utils.replay_buffer import ReplayBuffer

class DQNAgent():
    def __init__(self, input_shape, action_size, device, buffer_size, batch_size, gamma, lr, tau, update_every, replay_after, model):
        self.input_shape = input_shape
        self.action_size = action_size
        self.device = device
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.update_every = update_every
        self.replay_after = replay_after
        self.DQN = model
        self.tau = tau
        
        self.policy_net = self.DQN(input_shape, action_size).to(self.device)
        self.target_net = self.DQN(input_shape, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size, self.device)
        
        self.t_step = 0

    
    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        
        self.t_step = (self.t_step + 1) % self.update_every

        if self.t_step == 0:
            if len(self.memory) > self.replay_after:
                experiences = self.memory.sample()
                self.learn(experiences)
                
    def act(self, state, eps=0.):

        state = torch.from_numpy(state).unsqueeze(0).to(self.device)
        self.policy_net.eval()
        with torch.no_grad():
            action_values = self.policy_net(state)
        self.policy_net.train()
        
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
        
    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        Q_expected_current = self.policy_net(states)
        Q_expected = Q_expected_current.gather(1, actions.unsqueeze(1)).squeeze(1)

        Q_targets_next = self.target_net(next_states).detach().max(1)[0]
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        
        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.policy_net, self.target_net, self.tau)

    def soft_update(self, policy_model, target_model, tau):
        for target_param, policy_param in zip(target_model.parameters(), policy_model.parameters()):
            target_param.data.copy_(tau*policy_param.data + (1.0-tau)*target_param.data)

    def save_model(self, model, name):
        self.policy_net.save_model(path=('./Models/{}_{}_policy_model'.format(model, name)))
        self.target_net.save_model(path=('./Models/{}_{}_target_model'.format(model, name)))

    def load_model(self, model,name):
        self.policy_net.load_state_dict(torch.load('./Models/{}_{}_policy_model'.format(model, name)))
        self.target_net.load_state_dict(torch.load('./Models/{}_{}_target_model'.format(model, name)))

    def print_model_state_dict(self):
        print("Policy Network's state_dict:")
        for param_tensor in self.policy_net.state_dict():
            print(param_tensor, "\t", self.policy_net.state_dict()[param_tensor].size())        
    
        print("Target Network's state_dict:")
        for param_tensor in self.target_net.state_dict():
            print(param_tensor, "\t", self.target_net.state_dict()[param_tensor].size())