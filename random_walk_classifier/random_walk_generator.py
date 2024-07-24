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

    def generate_invalid_random_walks(self, valid_walks, corruption_prob=0.5):
        nodes = list(self.graph.nodes)
        edges = list(self.graph.edges(data=True))
        invalid_walks = []

        for walk in valid_walks:
            invalid_walk = walk.copy()
            for i in range(len(invalid_walk)):
                if random.random() < corruption_prob:
                    if i % 2 == 0:  # Change entity
                        invalid_walk[i] = random.choice(nodes)
                    else:  # Change relation
                        random_edge = random.choice(edges)
                        invalid_walk[i] = random_edge[2]['relation']
            invalid_walks.append(invalid_walk)

        return invalid_walks

