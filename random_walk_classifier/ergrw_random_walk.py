import networkx as nx
import random


class ERGRWRandomWalk:
    def __init__(self, graph: nx.Graph, alpha):
        self.G = graph
        self.alpha = alpha

    def random_walk_rule1(self, start_node, walk_length):
        """ Perform a Rule1 (entity-to-entity) walk step. """
        sequence = [start_node]
        current_node = start_node
        for _ in range(walk_length - 1):
            neighbors = list(self.G.neighbors(current_node))
            if not neighbors:
                break
            next_node = random.choice(neighbors)
            sequence.append(next_node)
            current_node = next_node
        return sequence

    def random_walk_rule2(self, start_node, walk_length):
        """ Perform a Rule2 (entity-relation) walk step. """
        sequence = [start_node]
        current_node = start_node
        for _ in range(walk_length // 2):
            relation_neighbors = list(self.G[current_node].keys())  # Get relation types (edge labels)
            if not relation_neighbors:
                break
            next_relation = random.choice(relation_neighbors)
            sequence.append(next_relation)

            entity_neighbors = list(self.G[current_node][next_relation])
            if not entity_neighbors:
                break
            next_node = random.choice(entity_neighbors)
            sequence.append(next_node)

            current_node = next_node
        return sequence

    def generate_walks(self, num_walks, walk_length, min_walk_length=3):
        """ Generate a specified number of random walks of a given length. """
        walks = []
        nodes = list(self.G.nodes)
        while len(walks) < num_walks:
            start_node = random.choice(nodes)
            if random.random() < self.alpha:
                walk = self.random_walk_rule1(start_node, walk_length)
            else:
                walk = self.random_walk_rule2(start_node, walk_length)

            # Ensure each walk generated has the minimum walk length
            if len(walk) >= min_walk_length:
                walks.append(walk)

        return walks
