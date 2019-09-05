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


def generate_maps(ball_count, buckets, width=10, height=10):
    """
        ball_count: dictionary of characters and number of occurence
    """
    assert isinstance(ball_count, dict), "ball_count must be dictionary"
    objects = deepcopy(ball_count)
    worldmap = []
    worldmap.append("#"*width)
    for row in range(height - 2):
        worldmap.append("#" + " "*(width - 2) + "#")
    worldmap.append("#"*width)

    empty_spaces = np.vstack([np.frombuffer(line.encode("ascii"),
                                            dtype=np.uint8)
                              for line in worldmap])
    height, width = empty_spaces.shape
    possible_locations = np.argwhere(empty_spaces.reshape(-1) == 32)
    n_objects = 1 + len(buckets) + sum(v for ball, v in objects.items())
    locations = np.random.choice(possible_locations.reshape(-1),
                                 size=n_objects, replace=False)
    index = 0
    objects["P"] = 1
    for bucket in buckets:
        objects[bucket] = 1
    for char, occurence in objects.items():
        for i in range(occurence):
            y = locations[index] // width
            x = locations[index] % width
            row = list(worldmap[y])
            row[x] = char
            worldmap[y] = "".join(row)
            index += 1
    return worldmap


def warehouse_setting(ball_count, balls, n_maps, pairing):
    assert isinstance(ball_count, dict)
    assert isinstance(balls, str)
    n_objects = 2 + len(pairing.keys()) + len(balls)
    n_edge = 3
    adj = torch.zeros(n_edge, n_objects, n_objects)
    objs = {"#": 0}
    for bucket in sorted(pairing.keys()):
        objs[bucket] = len(objs)
    objs["P"] = len(objs)
    for ball in sorted(balls):
        objs[ball] = len(objs)

    ordered_balls = {k: i for i, k in enumerate(sorted(balls))}
    ordered_buckets = {k: i for i, k in enumerate(sorted(pairing.keys()))}

    # "#, B, P, b, c, d"
    # Impassible edges
    adj[0][objs["P"], objs["#"]] = 1.0  # Player to wall
    for bucket in pairing.keys():
        adj[0][objs["P"], objs[bucket]] = 1.0
    # Collectable edges
    for ball in ball_count.keys():
        adj[1][objs["P"], objs[ball]] = 1.0  # Player to ball
    # Ball to bucket edges
    for bucket, balls in pairing.items():
        for ball in balls:
            adj[2][objs[ball], objs[bucket]] = 1.0

    worldmaps = [generate_maps(ball_count, pairing.keys())
                 for i in range(n_maps)]
    return worldmaps, pairing, lambda device: adj.to(device)


def run(network_class, index=0, test=False):
    balls = "abcde"
    buckets = "ABCDE"
    train_pairing = {"A": ["a"], "B": ["b"], "C": ["c"], "D": ["d"]}
    test_pairing = {"A": ["a"]}
    train_ball_counts = {"a": 1, "b": 1, "c": 1, "d": 1}
    test_ball_counts = {"a": 1}
    n_train_maps = 100
    n_test_maps = 1000
    dir_path = ("experiments/Relational_a2c_maxpool_concat_dropout_multi/" +
                str(index) + "/")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    model_path = dir_path + "param.b"
    hyperparam_path = dir_path + "hyperparam.yaml"

    hyperparams = dict(
        gamma=0.99,
        nenv=8,
        nstep=20,
        n_timesteps=200000,
        lr=0.0001,
        beta=0.03,
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
                                           network_class, render=True,
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

    NETWORK_CLASS = RelationalNet
    # processes = []
    # for i in range(1):
    #     process = torch.multiprocessing.Process(
    #         target=run, args=(NETWORK_CLASS, i, False))
    #     process.start()
    #     processes.append(process)

    # for p in processes:
    #     p.join()

    run(NETWORK_CLASS, index=0, test=True)
