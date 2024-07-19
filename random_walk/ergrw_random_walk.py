import numpy as np
import networkx as nx


class ERGRWRandomWalk:
    def __init__(self, G: nx.Graph, alpha=0.5):
        self.G = G
        self.alpha = alpha
        self.entities = list(G.nodes)
        self.relations = list(set(edge[2]['relation'] for edge in G.edges(data=True)))

    def rule1_walk(self, start_node, walk_length):
        walk = [start_node]
        for _ in range(walk_length):
            neighbors = list(self.G.neighbors(walk[-1]))
            if not neighbors:
                break
            next_node = np.random.choice(neighbors)
            walk.append(next_node)
        return walk

    def rule2_walk(self, start_node, walk_length):
        walk = [start_node]
        for _ in range(walk_length // 2):
            # Entity to Relation
            neighbors = [(nbr, self.G[start_node][nbr]['relation']) for nbr in self.G.neighbors(start_node)]
            if not neighbors:
                break
            next_node, relation = neighbors[np.random.choice(len(neighbors))]
            walk.extend([relation, next_node])
            start_node = next_node
        return walk

    def generate_walks(self, num_walks, walk_length):
        walks = []
        for _ in range(num_walks):
            for entity in self.entities:
                if np.random.rand() < self.alpha:
                    walks.append(self.rule1_walk(entity, walk_length))
                else:
                    walks.append(self.rule2_walk(entity, walk_length))
        return walks