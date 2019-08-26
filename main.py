import torch
import numpy as np
from copy import deepcopy
import yaml
import matplotlib.pyplot as plt
import os

from test import test_agent
from train import train_agent


def generate_maps(ball_count):
    """
        ball_count: dictionary of characters and number of occurence
    """
    assert isinstance(ball_count, dict), "ball_count must be dictionary"
    objects = deepcopy(ball_count)
    worldmap = ["##########",
                "#        #",
                "#        #",
                "#        #",
                "#        #",
                "#        #",
                "#        #",
                "#        #",
                "#        #",
                "##########"]
    empty_spaces = np.vstack([np.frombuffer(line.encode("ascii"),
                                            dtype=np.uint8)
                              for line in worldmap])
    height, width = empty_spaces.shape
    possible_locations = np.argwhere(empty_spaces.reshape(-1) == 32)
    n_objects = 2 + sum(v for ball, v in objects.items())
    locations = np.random.choice(possible_locations.reshape(-1),
                                 size=n_objects, replace=False)
    index = 0
    objects["P"] = 1
    objects["B"] = 1
    for char, occurence in objects.items():
        for i in range(occurence):
            y = locations[index] // width
            x = locations[index] % width
            row = list(worldmap[y])
            row[x] = char
            worldmap[y] = "".join(row)
            index += 1
    return worldmap


def warehouse_setting(ball_count, balls, n_maps, bucket="B"):
    assert isinstance(ball_count, dict)
    assert isinstance(balls, str)
    n_objects = 3 + len(balls)
    n_edge = 3
    adj = torch.zeros(n_edge, n_objects, n_objects)
    ordered_balls = {k: i for i, k in enumerate(sorted(balls))}

    # "#, B, P, b, c, d"
    # Impassible edges
    adj[0][2, 0] = 1.0  # Player to wall
    adj[0][2, 1] = 1.0  # Player to bucket
    # Collectable edges
    for ball in ball_count.keys():
        adj[1][2, ordered_balls[ball]+3] = 1.0  # Player to ball
    # Bucket to ball edges
    for ball in ball_count.keys():
        adj[2][ordered_balls[ball]+3, 1] = 1.0

    pairing = {
        bucket: [char for char in ball_count.keys()],
    }

    worldmaps = [generate_maps(ball_count) for i in range(n_maps)]
    return worldmaps, pairing, lambda device: adj.to(device)


def run(index=0, test=False):
    balls = "bcd"
    bucket = "B"
    train_ball_counts = {"b": 1}
    test_ball_counts = {"b": 1}
    n_train_maps = 100
    n_test_maps = 1000
    dir_path = "experiments/a2c_maxpool_concat/" + str(index) + "/"
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
        beta=0.11,
    )

    if not test:
        yaml.dump(hyperparams, open(hyperparam_path, "w"))

        train_worldmaps, train_pairing, train_adj = warehouse_setting(
            train_ball_counts, balls, n_train_maps, bucket=bucket)
        train_agent(train_worldmaps, balls, bucket,
                    train_pairing, train_adj, hyperparams,
                    model_path, suffix=str(index))

    if test:
        test_worldmaps, test_pairing, test_adj = warehouse_setting(
            test_ball_counts, balls, n_test_maps, bucket=bucket)
        results = test_agent(test_worldmaps, balls, bucket,
                             test_pairing, test_adj, model_path,
                             render=False, n_test=1000)
        plt.hist(results)
        plt.show()


if __name__ == "__main__":

    processes = []
    for i in range(3):
        process = torch.multiprocessing.Process(target=run, args=(i, False))
        process.start()
        processes.append(process)

    for p in processes:
        p.join()

    run(test=True)
