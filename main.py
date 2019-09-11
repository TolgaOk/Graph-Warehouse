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
from tools.config import Config






if __name__ == "__main__":
    NETWORK_CLASS = GraphDqnModel

    processes = []
    for i in range(3):
        process = torch.multiprocessing.Process(
            target=run, args=(NETWORK_CLASS, i, False))
        process.start()
        processes.append(process)

    for p in processes:
        p.join()

    run(NETWORK_CLASS, index=0, test=True)
