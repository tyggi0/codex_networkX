import networkx as nx
from codex.codex import Codex
from torch.utils.data import DataLoader
from random_walk.brownian_motion_random_walk import BrownianMotionRandomWalk
from random_walk.ergrw_random_walk import ERGRWRandomWalk
from random_walk.node2_vec_random_walk import Node2VecRandomWalk
from random_walk_classifier.random_walk_classifier import RandomWalkClassifier
import argparse


def get_random_walk_strategy(name, graph):
    if name == "BrownianMotion":
        return BrownianMotionRandomWalk(graph)
    elif name == "ERGRW":
        return ERGRWRandomWalk(graph)
    elif name == "Node2Vec":
        return Node2VecRandomWalk(graph)
    else:
        raise ValueError(f"Unknown random walk strategy: {name}")


def main(random_walk_name):
    # Initialize Codex
    codex = Codex(code="en", size="s")

    # Access entities, relations, and triples
    entities = codex.entities()
    relations = codex.relations()
    triples = codex.triples()

    print("Number of entities:", len(entities))
    print("Number of relations:", len(relations))
    print("Number of triples:", len(triples))

    # Access train, validation, and test triples
    train_triples = codex.split("train")
    valid_triples = codex.split("valid")
    test_triples = codex.split("test")

    # Construct graphs
    def construct_graph(triples):
        G = nx.Graph()

        # Add entities as nodes
        for entity in entities:
            G.add_node(entity)

        # Add triples as edges
        for head, relation, tail in triples:
            G.add_edge(head, tail, relation=relation)
        return G

    train_graph = construct_graph(train_triples)
    valid_graph = construct_graph(valid_triples)
    test_graph = construct_graph(test_triples)

    # Initialize classifier with the specified random walk strategy
    random_walk_strategy = get_random_walk_strategy(random_walk_name, train_graph)
    classifier = RandomWalkClassifier(train_graph, random_walk_strategy)

    # Generate walks
    valid_walks = classifier.generate_random_walks(num_walks=50, walk_length=5)
    invalid_walks = classifier.generate_invalid_random_walks(num_walks=50, walk_length=5)
    print("Valid Walks:")
    [print(walk) for walk in valid_walks]

    print("Invalid Walks:")
    [print(walk) for walk in invalid_walks]

    walks, labels = classifier.prepare_data(valid_walks, invalid_walks)
    dataset = RandomWalkClassifier.WalkDataset(walks, labels)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Train the model
    classifier.train(dataloader)

    # Generate test data
    random_walk_strategy = get_random_walk_strategy(random_walk_name, test_graph)
    classifier = RandomWalkClassifier(test_graph, random_walk_strategy)

    test_valid_walks = classifier.generate_random_walks(num_walks=20, walk_length=5)
    test_invalid_walks = classifier.generate_invalid_random_walks(num_walks=20, walk_length=5)

    test_walks, test_labels = classifier.prepare_data(test_valid_walks, test_invalid_walks)
    test_dataset = RandomWalkClassifier.WalkDataset(test_walks, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Evaluate the model
    classifier.evaluate(test_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Random Walk Classifier')
    parser.add_argument('--random_walk', type=str, required=True,
                        help='Name of the random walk strategy (BrownianMotion, ERGRW, Node2Vec)')
    args = parser.parse_args()
    main(args.random_walk)

    # Running script:
    # python main.py --random_walk BrownianMotion
    # python main.py --random_walk ERGRW
    # python main.py --random_walk Node2Vec
