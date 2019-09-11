import torch
import numpy as np
import random
import yaml
from copy import deepcopy

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

    @staticmethod
    def get_adjacency(ball_count, balls, pairing):
        """
        Return adjacency
        """
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

        return adj
    
    @staticmethod
    def prep_env(path):
        data = yaml.load(open(path,"r"))
        worldmaps = [VariationalWarehouse.generate_maps(data["ball_count"], 
                                                        data["buckets"], 
                                                        data["mapsize"], data["mapsize"]) 
                                                        for i in range(data["n_worlds"])]
        
        graph = VariationalWarehouse.get_adjacency(data["ball_count"], 
                                                   data["balls"], data["pairing"])
        n_objects = 2 + len(data["pairing"].keys()) + len(data["balls"])
        environment_kwargs = dict(
            graph = graph,
            in_channel = n_objects,
            out_channel = 4,
            mapsize = data["mapsize"],
            balls = data["balls"],
            buckets = data["buckets"],
            pairing = data["pairing"],
            worldmaps = worldmaps,
            )
        return environment_kwargs