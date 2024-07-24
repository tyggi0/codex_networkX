import networkx as nx
import random


class ERGRWRandomWalk:
    def __init__(self, graph: nx.Graph, alpha=0.5):
        self.G = graph
        self.alpha = alpha

    def rule1_walk(self, current):
        """ Perform a Rule1 (entity-to-entity) walk step. """
        neighbors = list(self.G.neighbors(current))
        if neighbors:
            return random.choice(neighbors)
        return None

    def rule2_walk(self, current):
        """ Perform a Rule2 (entity-relation) walk step. """
        neighbors = list(self.G.neighbors(current))
        if neighbors:
            next_node = random.choice(neighbors)
            relation = self.G[current][next_node]['relation']
            return relation, next_node
        return None, None

    def generate_walk(self, start_node, walk_length):
        """ Generate a single walk from a given start node. """
        walk = []
        current = start_node
        for _ in range(walk_length):
            if random.random() < self.alpha:
                # Rule1 walk step
                next_node = self.rule1_walk(current)
                if next_node:
                    walk.append(current)
                    current = next_node
                else:
                    break
            else:
                # Rule2 walk step
                relation, next_node = self.rule2_walk(current)
                if next_node:
                    walk.append(current)
                    walk.append(relation)
                    current = next_node
                else:
                    break
        if len(walk) < walk_length:
            walk.append(current)
        return walk

    def generate_walks(self, num_walks, walk_length):
        """ Generate a specified number of random walks of a given length. """
        walks = []
        nodes = list(self.G.nodes)
        for _ in range(num_walks):
            start_node = random.choice(nodes)
            walk = self.generate_walk(start_node, walk_length)
            walks.append(walk)
        return walks
