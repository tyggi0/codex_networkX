import random
from random_walk_classifier.traditional_random_walk import TraditionalRandomWalk
from random_walk_classifier.ergrw_random_walk import ERGRWRandomWalk
from random_walk_classifier.adapted_ergrw_random_walk import AdaptedERGRWRandomWalk


class RandomWalkGenerator:
    def __init__(self, graph):
        self.graph = graph

    def get_random_walk_strategy(self, name, alpha):
        if name == "traditional":
            return TraditionalRandomWalk(self.graph)
        elif name == "ergrw":
            return ERGRWRandomWalk(self.graph, alpha)
        elif name == "adapted_ergrw":
            return AdaptedERGRWRandomWalk(self.graph, alpha)
        else:
            raise ValueError(f"Unknown random walk strategy: {name}")

    def generate_random_walks(self, random_walk_strategy, alpha, num_walks, walk_length):
        # Identify high-degree nodes
        degree_dict = dict(self.graph.degree())
        # Sort nodes by degree in descending order
        sorted_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)
        sorted_nodes = sorted_nodes[:num_walks // 2]
        print(sorted_nodes)

        random_walk = self.get_random_walk_strategy(random_walk_strategy, alpha)
        return random_walk.generate_walks(sorted_nodes, 2, walk_length)

    def generate_invalid_random_walks(self, walks, corruption_prob=0.5):
        nodes = list(self.graph.nodes)
        edges = list(self.graph.edges(data=True))
        invalid_walks = []

        for walk in walks:
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

