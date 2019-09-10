import torch
import numpy as np
from copy import deepcopy
import yaml
import matplotlib.pyplot as plt
import os

from test import test_agent
from train import train_agent
from knowledgenet import GraphDqnModel
from relationalnet import RelationalNet
from vanillanet import ConvModel
from ourattnnet import OurAttnNet
from environments.warehouse import VariationalWarehouse




def run(network_class, index=0, test=False):
    ball_count = {"a": 1, "b": 1}
    balls = "bcd"
    buckets = "B"
    pairing = {"A": ["a"], "B": ["b"]}
    width=10
    height=10
    n_worlds = 100
    worldmaps = [VariationalWarehouse.generate_maps(ball_count,buckets,width,height) 
                for i in range(n_worlds)]
    
    graph = VariationalWarehouse.get_adjacency(ball_count, balls, pairing)
    environment_kwargs = dict(
        balls = balls,
        buckets = buckets,
        pairing = pairing,
        worldmaps = worldmaps
        )
    
    model_kwargs = dict(
        in_channel =  
        mapsize, 
        n_act, 
        n_entity=4, 
        n_heads=4
    )
    dir_path = ("experiments/OurAttnNet_a2c_maxpool_fourier/" +
                str(index) + "/")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    model_path = dir_path + "param.b"
    hyperparam_path = dir_path + "hyperparam.yaml"

    hyperparams = dict(
        gamma=0.99,
        nenv=8,
        nstep=20,
        n_timesteps=2000000,
        lr=0.0001,
        beta=0.06,
        attn_beta=0,
    )

    if not test:
        yaml.dump(hyperparams, open(hyperparam_path, "w"))

        train_worldmaps, train_pairing, train_adj = warehouse_setting(
            train_ball_counts, balls, n_train_maps, train_pairing)
        train_agent(train_worldmaps, balls, buckets,
                    train_pairing, train_adj, hyperparams,
                    network_class, model_path, suffix=str(index),
                    loadstates=False)

    if test:
        test_worldmaps, test_pairing, test_adj = warehouse_setting(
            test_ball_counts, balls, n_test_maps, test_pairing)
        results, success_list = test_agent(test_worldmaps, balls, buckets,
                                           test_pairing, test_adj, model_path,
                                           network_class, render=False,
                                           n_test=100)
        plt.subplot(211)
        plt.hist(results, bins=20, rwidth=0.30)
        plt.ylabel("reward")
        plt.title("Reward Histogram")
        plt.subplot(212)
        plt.hist(success_list, bins=2, rwidth=0.30)
        plt.ylabel("rate")
        plt.xlabel("Success rate: {0:.2f}".format(np.mean(success_list)))
        plt.title("Success Histogram")
        plt.show()


if __name__ == "__main__":
    NETWORK_CLASS = GraphDqnModel

    # processes = []
    # for i in range(3):
    #     process = torch.multiprocessing.Process(
    #         target=run, args=(NETWORK_CLASS, i, False))
    #     process.start()
    #     processes.append(process)

    # for p in processes:
    #     p.join()

    run(NETWORK_CLASS, index=0, test=True)
