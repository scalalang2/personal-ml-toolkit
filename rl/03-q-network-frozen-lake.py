# reference: https://www.kaggle.com/wuhao1542/pytorch-rl-0-frozenlake-q-network-learning
import gym
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
from torch.nn import functional as F 
import matplotlib.pyplot as plt

def one_hot(ids, nb_digits):
    if not isinstance(ids, (list, np.ndarray)):
        raise ValueError("ids must be 1-D list or array")

    batch_size = len(ids)
    ids = torch.LongTensor(ids).view(batch_size, 1)
    out_tensor = Variable(torch.FloatTensor(batch_size, nb_digits))
    out_tensor.data.zero_()
    out_tensor.data.scatter_(dim=1, index= ids, value=1)
    return out_tensor

def uniform_linear_layer(linear_layer):
    linear_layer.weight.data.uniform_()
    linear_layer.bias.data.fill_(-0.02)

lake = gym.make("FrozenLake-v0")
lake.reset()
lake.render()

# Define Agent model
class Agent(nn.Module):
    def __init__(self, observation_space_size, action_space_size):
        super(Agent, self).__init__()
        self.observation_space_size = observation_space_size
        self.hidden_size = observation_space_size
        self.l1 = nn.Linear(in_features=observation_space_size, out_features=self.hidden_size)
        self.l2 = nn.Linear(in_features=self.hidden_size, out_features=action_space_size)

        uniform_linear_layer(self.l1)
        uniform_linear_layer(self.l2)
    
    def forward(self, state):
        obs_emb = one_hot([int(state)], self.observation_space_size)
        out1 = torch.sigmoid(self.l1(obs_emb))
        return self.l2(out1).view((-1))

class Trainer:
    def __init__(self):
        self.agent = Agent(lake.observation_space.n, lake.action_space.n)
        self.optimizer = optim.Adam(params=self.agent.parameters())
        self.success = []
        self.jList = []

    def train(self, epoch):
        for step in range(epoch):
            s = lake.reset()
            j = 0
            while j < 200:
                a = self.choose_action(s)
                s1, r, d, _ = lake.step(a)
                if d == True and r == 0:
                    r = -1
                
                target_q = r + 0.99 * torch.max(self.agent(s1).detach())
                loss = F.smooth_l1_loss(self.agent(s)[a], target_q)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                s = s1
                j += 1
                if d == True: 
                    break
            
            if d == True and r > 0:
                self.success.append(1)
            else: 
                self.success.append(0)

            self.jList.append(j)
        
            if step % 100 == 0:
                print("last 100 epoches success rate: " + str(sum(self.success)/len(self.success)) + "%")

    def choose_action(self, s):
        if np.random.rand(1) < 0.1:
            return lake.action_space.sample()
        else:
            agent_out = self.agent(s).detach()
            _, max_index = torch.max(agent_out, 0)
            return max_index.data.item()

t = Trainer()
t.train(2000)

