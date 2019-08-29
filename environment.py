import torch
import numpy as np
import random

from gymcolab.envs.warehouse import Warehouse


class VariationalWarehouse(Warehouse):

    def __init__(self, balls, buckets, worldmaps, pairing):
        self.world_maps = worldmaps
        worldmap = random.sample(worldmaps, 1)[0]
        super().__init__(balls, buckets, worldmap=worldmap, pairing=pairing)

    def reset(self):
        self.world_map = random.sample(self.world_maps, 1)[0]
        return super().reset()
