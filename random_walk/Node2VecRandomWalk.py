from node2vec import Node2Vec

from random_walk.base_random_walk import BaseRandomWalk


class Node2VecRandomWalk(BaseRandomWalk):
    def __init__(self, dimensions=64, walk_length=30, num_walks=200, p=1, q=1, **kwargs):
        super().__init__()
        self.node2vec = Node2Vec(
            self.G, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, p=p, q=q, **kwargs
        )
        self.model = self.node2vec.fit()

    def walk(self, start_node, steps):
        return self.model.wv.most_similar(start_node, topn=steps)

    def generate_walks(self):
        # Use Node2Vec's built-in walk generation
        walks = self.node2vec.walks
        return walks