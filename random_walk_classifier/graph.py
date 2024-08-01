import networkx as nx


class Graph:
    def __init__(self, codex):
        self.codex = codex
        self.train_graph = None

    def construct_labeled_graph(self, triples):
        G = nx.DiGraph()
        for head, relation, tail in triples.values:
            head_label = self.codex.entity_label(head)
            relation_label = self.codex.relation_label(relation)
            tail_label = self.codex.entity_label(tail)
            G.add_edge(head_label, tail_label, relation=relation_label)
            # Solve no neighbour problem, add inverse relation
            G.add_edge(tail_label, head_label, relation="~" + relation_label)
        return G

    def load_graph(self):
        train_triples = self.codex.split("train")

        # Print details about the splits
        print("Train Split:")
        print(f"Number of triples: {len(train_triples)}")
        print(f"Head of triples:\n{train_triples.head()}\n")

        self.train_graph = self.construct_labeled_graph(train_triples)

        return self.train_graph
