import networkx as nx


class BaseRandomWalk:
    def __init__(self, G: nx.Graph, walk_length=6, num_walks=10):
        self.G = G
        self.walk_length = walk_length
        self.num_walks = num_walks

    def generate_walks(self):
        raise NotImplementedError("This method should be overridden by subclasses")
