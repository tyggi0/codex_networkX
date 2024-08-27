import random
from traditional_random_walk import TraditionalRandomWalk
from ergrw_random_walk import ERGRWRandomWalk
from adapted_ergrw_random_walk import AdaptedERGRWRandomWalk


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

    def generate_random_walks(self, random_walk_strategy, alpha, num_walks, walk_length, mode):
        random_walk = self.get_random_walk_strategy(random_walk_strategy, alpha)

        # Identify high-degree nodes
        degree_dict = dict(self.graph.degree())
        # Sort nodes by degree in descending order
        sorted_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)

        # Calculate the number of walks per node dynamically
        num_nodes = len(sorted_nodes)
        if num_nodes > 0:
            walks_per_node = max(1, num_walks // num_nodes)
        else:
            walks_per_node = num_walks  # fallback if no nodes are found

        # Adjust the number of sorted nodes if necessary
        if num_nodes * walks_per_node > num_walks:
            num_nodes = num_walks // walks_per_node

        # Limit the sorted_nodes to the top `num_nodes`
        sorted_nodes = sorted_nodes[:num_nodes]

        # Generate the walks
        generated_walks = random_walk.generate_walks(sorted_nodes, walks_per_node, walk_length)

        # Check if we generated enough walks
        if len(generated_walks) < num_walks:
            additional_walks_needed = num_walks - len(generated_walks)
            additional_walks = random_walk.generate_walks(sorted_nodes, walks_per_node + 1, walk_length)
            generated_walks.extend(additional_walks[:additional_walks_needed])

        return generated_walks


    def generate_invalid_random_walks(self, walks, corruption_prob=0.4):
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
                        invalid_walk[i] = random_edge[2]['key']
            invalid_walks.append(invalid_walk)

        return invalid_walks

