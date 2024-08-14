import networkx as nx
import random


class AdaptedERGRWRandomWalk:
    def __init__(self, graph: nx.Graph, alpha):
        self.G = graph
        self.alpha = alpha

    def rule1_walk(self, neighbors):
        """ Perform a Rule1 (entity-to-entity) walk step, including the 'is connected to' relationship. """
        if neighbors:
            next_node = random.choice(neighbors)
            return "is connected to", next_node
        return None, None

    def rule2_walk(self, current, neighbors):
        """ Perform a Rule2 (entity-relation) walk step. """
        if neighbors:
            next_node = random.choice(neighbors)
            relation = self.G[current][next_node].get('key', None)
            return relation, next_node
        return None, None

    def generate_walk(self, start_node, walk_length):
        """ Generate a single walk from a given start node. """
        walk = [start_node]
        current = start_node
        step = 0

        while step < walk_length - 1:
            neighbors = list(self.G.neighbors(current))
            if not neighbors:
                break

            if random.random() < self.alpha:
                # Apply Rule1 (entity-to-entity)
                relation, next_node = self.rule1_walk(neighbors)
            else:
                # Apply Rule2 (entity-relation)
                relation, next_node = self.rule2_walk(current, neighbors)

            if next_node:
                walk.append(relation)
                walk.append(next_node)
                current = next_node
                step += 2
            else:
                continue  # If no valid move, attempt another step
        return walk

    def generate_walks(self, nodes, num_walks, walk_length, min_walk_length=3):
        """ Generate a specified number of random walks of a given length. """
        walks = []
        for node in nodes:
            for _ in range(num_walks):
            walk = self.generate_walk(node, walk_length)

            if len(walk) >= min_walk_length:
                walks.append(walk)
        return walks
