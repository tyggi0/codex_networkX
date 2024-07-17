import networkx as nx
from codex.codex import Codex

codex = Codex(code="en", size="s")

# Access entities, relations, and triples
entities = codex.entities()
relations = codex.relations()
triples = codex.triples()

# Print some information
print("Number of entities:", len(entities))
print("Number of relations:", len(relations))
print("Number of triples:", len(triples))

# Access train, validation, and test triples
train_triples = codex.split("train")
valid_triples = codex.split("valid")
test_triples = codex.split("test")

# Print some examples
print("First few training triples:")
print(train_triples.head())
print("First few validation triples:")
print(valid_triples.head())
print("First few test triples:")
print(test_triples.head())

# Initialize a directed graph
G = nx.DiGraph()

# Add entities as nodes
for entity in entities:
    G.add_node(entity)

# Add triples as edges
for _, row in train_triples.iterrows():
    head, relation, tail = row
    G.add_edge(head, tail, relation=relation)

