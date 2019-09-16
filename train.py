import torch
import numpy as np
import gym
import argparse
import logging
import lamp

from rl_pysc2.agents.a2c.model import A2C
from rl_pysc2.utils.parallel_envs import ParallelEnv
from graph_rl.environments.warehouse import VariationalWarehouse
from graph_rl.models.knowledgenet import GraphA2C
from graph_rl.models.ourattnnet import OurAttnNet
from graph_rl.models.relationalnet import RelationalNet
from graph_rl.models.vanillanet import ConvModel
from graph_rl.tools.config import Config


def train_agent(load_name, resume=False, forced=False, suffix='0', **residual):
    config = Config.load(load_name, overwrite=True)
    if not forced and config.model_params:
        raise RuntimeError(
            "Config file already occupied. Force it to overwrite")
    logger = configure_logger(config.logger_config)
    device = config.hyperparams['device']
    env = config.initiate_env()
    network = config.initiate_model()
    optimizer = torch.optim.Adam(network.parameters(),
                                 lr=config.hyperparams["lr"])
    agent = A2C(network, optimizer)
    agent.to(device)
    loss = 0

    penv = ParallelEnv(config.hyperparams["nenv"], config.initiate_env)
    eps_rewards = np.zeros((config.hyperparams["nenv"], 1))
    reward_list = [0]
    success_list = [0]

    def to_torch(array):
        return torch.from_numpy(array).to(device).float()
    logger.hyperparameters(config.hyperparams, win="Hyperparameters")

    if resume and config.model_params:
        agent.load_state_dict(config.model_params['agent'])
        optimizer.load_state_dict(config.model_params['optimizer'])

    with penv as state:
        state = to_torch(state)
        for i in range(config.hyperparams["n_timesteps"] //
                       config.hyperparams["nstep"]):
            for j in range(config.hyperparams["nstep"]):
                action, log_prob, value, entropy = agent(state)
                entropy = entropy
                action = action.unsqueeze(1).cpu().numpy()
                next_state, reward, done = penv.step(action)
                next_state = to_torch(next_state)
                with torch.no_grad():
                    _, next_value = agent.network(next_state)
                agent.add_trans(to_torch(reward), to_torch(done),
                                log_prob.unsqueeze(1), value,
                                next_value, entropy)
                state = next_state
                for j, d in enumerate(done.flatten()):
                    eps_rewards[j] += reward[j].item()
                    if d == 1:
                        success_list.append(float(reward[j] != -0.1))
                        reward_list.append(eps_rewards[j].item())
                        eps_rewards[j] = 0
                        logger.scalar(np.mean(reward_list[-10:]),
                                      env="main",
                                      win="reward_"+suffix, trace="Last 10")
                        logger.scalar(np.mean(reward_list[-50:]),
                                      env="main",
                                      win="reward_"+suffix, trace="Last 50")
                        logger.scalar(np.mean(success_list[-10:]),
                                      env="main",
                                      win="success_"+suffix, trace="Last 10")
                        logger.scalar(np.mean(success_list[-50:]),
                                      env="main",
                                      win="success_"+suffix, trace="Last 50")
                        logger.scalar(loss, env="main",
                                      win="loss_"+suffix)
                        # ---------------- Statistics ----------------------
                        # for key, stat in agent.network.attn_module.visual_attn.statistics.items():
                        #     trace_name, plot_type = key.split("_")
                        #     logger.scalar(stat,
                        #                   env="main",
                        #                   win=plot_type+suffix,
                        #                   trace=trace_name)
            loss = agent.update(
                config.hyperparams["gamma"], config.hyperparams["beta"])
            if i % 100 == 0:
                config.model_params = dict(agent=agent.state_dict(),
                                           optimizer=optimizer.state_dict())
                config.save(load_name, suffix=suffix)


def configure_logger(config):
    import logging.config
    logging.config.dictConfig(config)
    logger = logging.getLogger(__name__)
    return logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument("--name", help="Config file name to load",
                        action="store", dest="load_name")
    parser.add_argument(
        "--forced", help="force to overwrite", action='store_true')
    parser.add_argument(
        "--resume", help="initiates parameters from the given config",
        action='store_true')
    parser.add_argument("--n-train", help="number of trains",
                        action='store', type=int, dest='n_process', default=1)

    kwargs = vars(parser.parse_args())
    kwargs['load_name'] = "configs/configs/" + kwargs['load_name']
    processes = []
    for i in range(kwargs['n_process']):
        kwargs['suffix'] = str(i)
        process = torch.multiprocessing.Process(
            target=train_agent, kwargs=kwargs)
        process.start()
        processes.append(process)

    for p in processes:
        p.join()
