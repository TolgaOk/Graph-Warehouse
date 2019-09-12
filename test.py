import torch
import numpy as np
import time
import argparse

from environments.warehouse import VariationalWarehouse
from rl_pysc2.agents.a2c.model import A2C
from models.knowledgenet import GraphA2C
from tools.config import Config


def test_agent(load_name, render=True, n_test=1):
    config = Config.load(load_name, overwrite=True)
    device = config.hyperparams['device']

    env = config.initiate_env()
    network = config.initiate_model()
    optimizer = torch.optim.Adam(network.parameters(),
                                 lr=config.hyperparams["lr"])
    agent = A2C(network, optimizer)
    agent.to(device)
    agent.eval()

    reward_list = [0]
    success_list = [0]

    agent.load_state_dict(config.model_params['agent'])
    optimizer.load_state_dict(config.model_params['optimizer'])

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument("--name", help="Config file name to load",
                        action="store", dest="load_name")
    parser.add_argument("--render", help="to render", action='store_true')
    parser.add_argument("--iter", help="number of tests",
                        action='store', type=int, dest='n_test', default=1)

    kwargs = vars(parser.parse_args())
    kwargs['load_name'] = "configs/configs/" + kwargs['load_name']
    test_agent(**kwargs)
