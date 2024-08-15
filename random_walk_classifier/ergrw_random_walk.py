import networkx as nx
import random


class ERGRWRandomWalk:
    def __init__(self, graph: nx.Graph, alpha):
        self.G = graph
        self.alpha = alpha

    def random_walk_rule1(self, start_node, walk_length):
        """ Perform a Rule1 (entity-to-entity) walk step. """
        walk = [start_node]
        current_node = start_node
        for _ in range(walk_length - 1):
            neighbors = list(self.G.neighbors(current_node))
            if not neighbors:
                break
            next_node = random.choice(neighbors)
            walk.append("is connected to")
            walk.append(next_node)
            current_node = next_node
        return walk

    def random_walk_rule2(self, start_node, walk_length):
        """ Perform a Rule2 (entity-relation) walk step. """
        walk = [start_node]
        current_node = start_node
        for _ in range(walk_length - 1):
            neighbors = list(self.G[current_node])
            if not neighbors:
                break
            next_node = random.choice(neighbors)
            next_relation = self.G[current_node][next_node]['key']
            if next_relation:
                walk.append(next_relation)
                walk.append(next_node)
                current_node = next_node
            else:
                break
        return walk

    def generate_walks(self, nodes, num_walks, walk_length, min_walk_length=3):
        """ Generate the specified number of random walks for each node. """
        walks = []
        for node in nodes:
            for _ in range(num_walks):
                if random.random() < self.alpha:
                    walk = self.random_walk_rule1(node, walk_length)
                else:
                    walk = self.random_walk_rule2(node, walk_length)

                # Ensure each walk generated has the minimum walk length
                if len(walk) >= min_walk_length:
                    walks.append(walk)
        return walks
