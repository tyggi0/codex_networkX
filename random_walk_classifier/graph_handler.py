import networkx as nx


class GraphHandler:
    def __init__(self, codex):
        self.codex = codex
        self.train_graph = None
        self.valid_graph = None
        self.test_graph = None

    def construct_labeled_graph(self, triples):
        G = nx.DiGraph()
        for head, relation, tail in triples.values:
            head_label = self.codex.entity_label(head)
            relation_label = self.codex.relation_label(relation)
            tail_label = self.codex.entity_label(tail)
            G.add_edge(head_label, tail_label, relation=relation_label)
        return G

    def load_graphs(self):
        train_triples = self.codex.split("train")
        valid_triples = self.codex.split("valid")
        test_triples = self.codex.split("test")

        # Print details about the splits
        print("Train Split:")
        print(f"Number of triples: {len(train_triples)}")
        print(f"Head of triples:\n{train_triples.head()}\n")

        print("Validation Split:")
        print(f"Number of triples: {len(valid_triples)}")
        print(f"Head of triples:\n{valid_triples.head()}\n")

        print("Test Split:")
        print(f"Number of triples: {len(test_triples)}")
        print(f"Head of triples:\n{test_triples.head()}\n")

        self.train_graph = self.construct_labeled_graph(train_triples)
        self.valid_graph = self.construct_labeled_graph(valid_triples)
        self.test_graph = self.construct_labeled_graph(test_triples)

        return self.train_graph, self.valid_graph, self.test_graph
