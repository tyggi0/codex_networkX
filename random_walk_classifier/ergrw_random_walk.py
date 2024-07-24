import networkx as nx
import random


class ERGRWRandomWalk:
    def __init__(self, graph: nx.Graph, alpha=0.5):
        self.G = graph
        self.alpha = alpha

    def rule1_walk(self, current, neighbors):
        """ Perform a Rule1 (entity-to-entity) walk step. """
        if neighbors:
            return random.choice(neighbors)
        return None

    def rule2_walk(self, current, neighbors):
        """ Perform a Rule2 (entity-relation) walk step. """
        if neighbors:
            next_node = random.choice(neighbors)
            relation = self.G[current][next_node]['relation']
            return relation, next_node
        return None, None

    def generate_walk(self, start_node, walk_length):
        """ Generate a single walk from a given start node. """
        # print(f"Starting walk from: {start_node}")
        walk = [start_node]
        current = start_node
        step = 0

        while step < walk_length - 1:
            neighbors = list(self.G.neighbors(current))
            if not neighbors:
                # print(f"No neighbors found for {current}. Ending walk early.")
                break

            if random.random() < self.alpha:
                # Apply Rule1 (entity-to-entity)
                next_node = self.rule1_walk(current, neighbors)
                if next_node:
                    # print(f"Rule1: Moving from {current} to {next_node}")
                    walk.append(next_node)
                    current = next_node
                    step += 1
                else:
                    # print(f"Rule1: No valid moves from {current}, retrying...")
                    continue  # If no valid move, attempt another step
            else:
                # Apply Rule2 (entity-relation)
                relation, next_node = self.rule2_walk(current, neighbors)
                if next_node:
                    # print(f"Rule2: Moving from {current} via {relation} to {next_node}")
                    walk.append(relation)
                    walk.append(next_node)
                    current = next_node
                    step += 2
                else:
                    # print(f"Rule2: No valid moves from {current}, retrying...")
                    continue  # If no valid move, attempt another step

        # print(f"Completed walk: {walk}")
        return walk

    def generate_walks(self, num_walks, walk_length):
        """ Generate a specified number of random walks of a given length. """
        walks = []
        nodes = list(self.G.nodes)
        for i in range(num_walks):
            start_node = random.choice(nodes)
            # print(f"Generating walk {i + 1}/{num_walks} from {start_node}")
            walk = self.generate_walk(start_node, walk_length)
            walks.append(walk)
        return walks
