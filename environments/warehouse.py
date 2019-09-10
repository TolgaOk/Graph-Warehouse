import torch
import numpy as np
import random

from gymcolab.envs.warehouse import Warehouse


class VariationalWarehouse(Warehouse):
    """ Randomly sample a worldmap from the given list of maps at each reset.
    Arguments:
        balls: String of ballrac characters
        buckets: String of bucket characters
        worldmaps: List of Warehouse maps that will be sampled at each reset
        pairing: Dictionary of ball bucket pairings
    """

    def __init__(self, balls, buckets, worldmaps, pairing):
        self.world_maps = worldmaps
        worldmap = random.sample(worldmaps, 1)[0]
        super().__init__(balls, buckets, worldmap=worldmap, pairing=pairing)

    def reset(self):
        self.world_map = random.sample(self.world_maps, 1)[0]
        return super().reset()

    @staticmethod
    def generate_maps(ball_count, buckets, width=10, height=10):
        """
        Generate random maps for the warehouse environments.
            Arguments:
                - ball_count: Dictionary of characters and number of occurence
                - buckets: String of bucket characters. Only a single instance
                for each buckets
                - width: Width of the map
                - height: Height of the map
            Return:
                A map with randomly placed objects
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
