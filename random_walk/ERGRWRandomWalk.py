import numpy as np
import networkx as nx

from random_walk.base_random_walk import BaseRandomWalk


class ERGRWRandomWalk(BaseRandomWalk):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.entities = list(G.nodes)
        self.relations = list(set(edge[2]['relation'] for edge in G.edges(data=True)))

    def rule1_walk(self, start_node):
        walk = [start_node]
        for _ in range(self.walk_length):
            neighbors = list(self.G.neighbors(walk[-1]))
            if not neighbors:
                break
            next_node = np.random.choice(neighbors)
            walk.append(next_node)
        return walk

    def rule2_walk(self, start_node):
        walk = [start_node]
        for _ in range(self.walk_length // 2):
            # Entity to Relation
            neighbors = [(nbr, self.G[start_node][nbr]['relation']) for nbr in self.G.neighbors(start_node)]
            if not neighbors:
                break
            next_node, relation = neighbors[np.random.choice(len(neighbors))]
            walk.extend([relation, next_node])
            start_node = next_node
        return walk

    def generate_walks(self):
        walks = []
        for _ in range(self.num_walks):
            for entity in self.entities:
                if np.random.rand() < self.alpha:
                    walks.append(self.rule1_walk(entity))
                else:
                    walks.append(self.rule2_walk(entity))
        return walks

    # def encode_walks(self, walks):
    #     # Assuming use of a BiLSTM encoder; this is a placeholder
    #     # Encode the walks to capture sequential information
    #     encoded_walks = []
    #     for walk in walks:
    #         encoded_walk = self.encode_walk(walk)
    #         encoded_walks.append(encoded_walk)
    #     return encoded_walks
    #
    # def encode_walk(self, walk):
    #     # Placeholder for actual BiLSTM encoding
    #     return walk
    #
    # def decode(self, encoded_walks):
    #     # Use the InteractE model for decoding; this is a placeholder
    #     decoded_results = []
    #     for walk in encoded_walks:
    #         decoded_result = self.interacte_decode(walk)
    #         decoded_results.append(decoded_result)
    #     return decoded_results
    #
    # def interacte_decode(self, walk):
    #     # Placeholder for actual InteractE decoding
    #     return walk

# Example usage
G = nx.MultiDiGraph()
G.add_edge('A', 'B', relation='friend')
G.add_edge('B', 'C', relation='colleague')
G.add_edge('C', 'D', relation='friend')
G.add_edge('A', 'D', relation='neighbor')

ergrw = ERGRW(G)
walks = ergrw.generate_walks()
encoded_walks = encode_walks(walks)
decoded_results = ergrw.decode(encoded_walks)

print("Generated Walks:")
for walk in walks:
    print(walk)

print("Encoded Walks:")
for e_walk in encoded_walks:
    print(e_walk)

print("Decoded Results:")
for d_result in decoded_results:
    print(d_result)
