import networkx as nx
import numpy as np


class TraditionalRandomWalk:
    def __init__(self, G: nx.Graph):
        self.G = G

    def walk(self, start_node, steps):
        path = [start_node]
        current_node = start_node

        for _ in range(steps):
            neighbors = list(self.G.neighbors(current_node))
            if not neighbors:
                break  # If there are no neighbors, break out of the loop
            next_node = np.random.choice(neighbors)
            path.append(next_node)
            current_node = next_node

        return path

    def generate_walks(self, num_walks, walk_length):
        walks = []
        nodes = list(self.G.nodes)

        for _ in range(num_walks):
            start_node = np.random.choice(nodes)
            walks.append(self.walk(start_node, walk_length))

        return walks
