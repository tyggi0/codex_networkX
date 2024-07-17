import networkx as nx
import random


class BaseRandomWalk:
    def __init__(self, G: nx.Graph, walk_length=6, num_walks=10):
        self.G = G
        self.walk_length = walk_length
        self.num_walks = num_walks

    def generate_walks(self):
        raise NotImplementedError("This method should be overridden by subclasses")

    def generate_invalid_walks(self):
        nodes = list(self.G.nodes)
        invalid_walks = []
        for _ in range(self.num_walks):
            walk = []
            current_node = random.choice(nodes)
            for _ in range(self.walk_length):
                walk.append(current_node)
                # Introduce invalid steps randomly
                if random.random() > 0.5:
                    current_node = random.choice(nodes)  # Random node, not necessarily a neighbor
                else:
                    neighbors = list(self.G.neighbors(current_node))
                    if neighbors:
                        current_node = random.choice(neighbors)
                    else:
                        break
            invalid_walks.append(walk)
        return invalid_walks
