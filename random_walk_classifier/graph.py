import networkx as nx


class Graph:
    def __init__(self, codex):
        self.codex = codex
        self.train_graph = None

    def create_labeled_graph(self, triples):
        G = nx.DiGraph()
        for head, relation, tail in triples.values:
            head_label = self.codex.entity_label(head)
            relation_label = self.codex.relation_label(relation)
            tail_label = self.codex.entity_label(tail)
            G.add_edge(head_label, tail_label, key=relation_label)
            # Solve no neighbour problem, add inverse relation
            G.add_edge(tail_label, head_label, key="~" + relation_label)
        return G

    def load_graph(self):
        train_triples = self.codex.split("train")

        # Print details about the splits
        print("Train Split:")
        print(f"Number of triples: {len(train_triples)}")
        print(f"Head of triples:\n{train_triples.head()}\n")

        self.train_graph = self.create_labeled_graph(train_triples)

        return self.train_graph


def create_sample_graph():
    """Create a sample graph directly, without using codex or triples."""
    G = nx.DiGraph()
    # Adding nodes
    G.add_node('A')
    G.add_node('B')
    G.add_node('C')
    G.add_node('D')
    G.add_node('E')

    # Adding edges with key types
    G.add_edge('A', 'B', key='likes')
    G.add_edge('B', 'C', key='friends with')
    G.add_edge('C', 'D', key='colleague of')
    G.add_edge('D', 'E', key='neighbor of')
    G.add_edge('E', 'A', key='family with')
    G.add_edge('A', 'C', key='knows')
    G.add_edge('B', 'D', key='works with')
    return G
