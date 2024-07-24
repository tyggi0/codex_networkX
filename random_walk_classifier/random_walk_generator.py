import random
from brownian_motion_random_walk import BrownianMotionRandomWalk
from ergrw_random_walk import ERGRWRandomWalk


class RandomWalkGenerator:
    def __init__(self, graph, random_walk_strategy):
        self.graph = graph
        self.random_walk = self.get_random_walk_strategy(random_walk_strategy)

    def get_random_walk_strategy(self, name):
        if name == "BrownianMotion":
            return BrownianMotionRandomWalk(self.graph)
        elif name == "ERGRW":
            return ERGRWRandomWalk(self.graph)
        else:
            raise ValueError(f"Unknown random walk strategy: {name}")

    def generate_random_walks(self, num_walks, walk_length):
        return self.random_walk.generate_walks(num_walks, walk_length)

    def generate_invalid_random_walks(self, num_walks, walk_length):
        nodes = list(self.graph.nodes)
        invalid_walks = []
        for _ in range(num_walks):
            walk = []
            current_node = random.choice(nodes)
            for _ in range(walk_length):
                walk.append(current_node)
                if random.random() > 0.5:
                    current_node = random.choice(nodes)
                else:
                    neighbors = list(self.graph.neighbors(current_node))
                    if neighbors:
                        current_node = random.choice(neighbors)
                    else:
                        break
            invalid_walks.append(walk)
        return invalid_walks

