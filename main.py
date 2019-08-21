import torch
import numpy as np
import gym

from rl_pysc2.agents.a2c.model import A2C
from rl_pysc2.utils.parallel_envs import ParallelEnv
from gymcolab.envs.warehouse import Warehouse
from vanillanet import ConvModel
from relationalnet import RelationalNet


def train():
    env = Warehouse()
    # env.settings['visualize'] = True
    logger = logger_config()

    hyperparams = dict(
        gamma=0.99,
        nenv=8,
        nstep=20,
        n_timesteps=100000,
        lr=0.0001,
    )

    env = Warehouse()
    in_channel, mapsize, _ = env.observation_space.shape
    n_act = 4
    network = RelationalNet(in_channel, mapsize, n_act)
    env.close()
    del env
    optimizer = torch.optim.RMSprop(network.parameters(),
                                    lr=hyperparams["lr"])
    agent = A2C(network, optimizer)
    device = "cuda"
    agent.to(device)
    loss = 0

    penv = ParallelEnv(hyperparams["nenv"], Warehouse)
    eps_rewards = np.zeros((hyperparams["nenv"], 1))
    reward_list = [0]

    def to_torch(array):
        return torch.from_numpy(array).to(device).float()
    logger.hyperparameters(hyperparams, win="Hyperparameters")

    with penv as state:
        state = to_torch(state)
        for i in range(hyperparams["n_timesteps"]//hyperparams["nstep"]):
            for j in range(hyperparams["nstep"]):
                action, log_prob, value = agent(state)
                action = action.unsqueeze(1).cpu().numpy()
                next_state, reward, done = penv.step(action)
                next_state = to_torch(next_state)
                with torch.no_grad():
                    _, next_value = agent.network(next_state)
                agent.add_trans(to_torch(reward), to_torch(done),
                                log_prob.unsqueeze(1), value,
                                next_value)
                state = next_state
                for j, d in enumerate(done.flatten()):
                    eps_rewards[j] += reward[j].item()
                    if d == 1:
                        reward_list.append(eps_rewards[j].item())
                        eps_rewards[j] = 0
                        logger.scalar(np.mean(reward_list[-20:]),
                                      win="reward", trace="Last 20")
                        logger.scalar(np.mean(reward_list[-50:]),
                                      win="reward", trace="Last 50")
                        logger.scalar(loss, win="loss")
                    print(("Epsiode: {}, Reward: {}, Loss: {}")
                          .format(len(reward_list)//hyperparams["nenv"],
                                  np.mean(reward_list[-100:]), loss),
                          end="\r")
            loss = agent.update(hyperparams["gamma"])
            if i % 10 == 0:
                agent.save_model("model_params/model_parameters.p")


def logger_config():
    import yaml
    import logging.config
    with open('logger_config.yaml', 'r') as f:
        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    logger = logging.getLogger(__name__)
    return logger


if __name__ == "__main__":
    train()
