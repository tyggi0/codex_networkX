import unittest
import networkx as nx
from random_walk.node2vec_random_walk import Node2VecRandomWalk


class TestNode2VecRandomWalk(unittest.TestCase):
    def test_walk(self):
        G = nx.erdos_renyi_graph(10, 0.5, seed=42)
        node2vec_rw = Node2VecRandomWalk(G)
        walks = node2vec_rw.generate_walks(num_walks=5, walk_length=5)
        for walk in walks:
            self.assertTrue(len(walk) > 0)


if __name__ == "__main__":
    unittest.main()
