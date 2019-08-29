import torch
import numpy as np
import time

from environment import VariationalWarehouse
from rl_pysc2.agents.a2c.model import A2C
from knowledgenet import GraphDqnModel


def test_agent(worldmaps, balls, buckets, relations,
               adjacency, load_param_path,
               network_class, render=True, n_test=1):
    device = "cuda"

    env = VariationalWarehouse(balls, buckets,
                               worldmaps=worldmaps, pairing=relations)
    in_channel, mapsize, _ = env.observation_space.shape
    n_act = 4
    if network_class == GraphDqnModel:
        adj = adjacency(device)
        network = network_class(adj.shape[0], in_channel, mapsize, n_act, adj)
    else:
        network = network_class(in_channel, mapsize, n_act)
    agent = A2C(network, None)
    agent.to(device)
    agent.eval()

    reward_list = [0]
    success_list = [0]

    agent.load_model(load_param_path)

    def to_torch(array):
        return torch.from_numpy(array).to(device).float()

    average_reward = []
    for i in range(n_test):
        eps_reward = 0
        state = env.reset()
        done = False
        while not done:
            if render:
                env.render()
                time.sleep(0.1)
            state = to_torch(state)
            action, log_prob, value, entropy = agent(state.unsqueeze(0))
            action = action.item()
            state, reward, done, _ = env.step(action)
            if done is True:
                success_list.append(float(reward != -0.1))
            eps_reward += reward
        average_reward.append(eps_reward)
        print("Progress: {}%".format(i/n_test*100), end="\r")

    return average_reward, success_list
