import networkx as nx
from codex.codex import Codex
import torch
from torch.utils.data import DataLoader
from random_walk_classifier.random_walk_classifier import RandomWalkClassifier
import argparse
import sys


def construct_labeled_graph(triples, codex):
    G = nx.Graph()
    for head, relation, tail in triples.values:
        head_label = codex.entity_label(head)
        relation_label = codex.relation_label(relation)
        tail_label = codex.entity_label(tail)
        G.add_edge(head_label, tail_label, relation=relation_label)
    return G


def main(random_walk_name, tune=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize Codex
    codex = Codex(code="en", size="s")

    # Access entities, relations, and triples
    entities = codex.entities()
    relations = codex.relations()
    triples = codex.triples()

    sys.stdout.write(f"Number of entities: {len(entities)}\n")
    sys.stdout.write(f"Number of relations: {len(relations)}\n")
    sys.stdout.write(f"Number of triples: {len(triples)}\n")

    # Access train, validation, and test triples
    train_triples = codex.split("train")
    valid_triples = codex.split("valid")
    test_triples = codex.split("test")

    sys.stdout.write("Constructing labeled graphs...\n")
    # Construct graphs with labeled triples
    train_graph = construct_labeled_graph(train_triples, codex)
    valid_graph = construct_labeled_graph(valid_triples, codex)
    test_graph = construct_labeled_graph(test_triples, codex)
    sys.stdout.write("Graphs constructed successfully.\n")

    # Initialize classifier with the specified random walk strategy
    classifier = RandomWalkClassifier(train_graph, random_walk_name, device=device)

    # Generate train walks
    sys.stdout.write("Generating train walks...\n")
    train_valid_walks = classifier.generate_random_walks(num_walks=500, walk_length=5)
    train_invalid_walks = classifier.generate_invalid_random_walks(num_walks=500, walk_length=5)
    sys.stdout.write(
        f"Generated {len(train_valid_walks)} valid train walks and {len(train_invalid_walks)} invalid train walks.\n")
    sys.stdout.write(f"Valid train walks: {train_valid_walks}\n")
    sys.stdout.write(f"Invalid train walks: {train_invalid_walks}\n")

    train_dataset = classifier.prepare_data(train_valid_walks, train_invalid_walks)

    valid_valid_walks = classifier.generate_random_walks(num_walks=100, walk_length=5)
    valid_invalid_walks = classifier.generate_invalid_random_walks(num_walks=100, walk_length=5)
    valid_dataset = classifier.prepare_data(valid_valid_walks, valid_invalid_walks)

    if tune:
        sys.stdout.write("Tuning hyperparameters...\n")
        best_config = classifier.tune_hyperparameters(train_dataset, valid_dataset)
        sys.stdout.write("Hyperparameter tuning completed.\n")

        # Set best hyperparameters and re-initialize the model and optimizer
        classifier.set_hyperparameters(
            learning_rate=best_config["learning_rate"],
            batch_size=best_config["batch_size"],
            epochs=best_config["epochs"]
        )

        # Train the model with the best hyperparameters
        train_dataloader = classifier.get_dataloader(train_dataset)
        sys.stdout.write("Training the model with best hyperparameters...\n")
        classifier.train(train_dataloader, epochs=best_config['epochs'])
        sys.stdout.write("Training completed.\n")
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

        # Train the model
        sys.stdout.write("Training the model...\n")
        classifier.train(train_dataloader)
        sys.stdout.write("Training completed.\n")

    # Initialize classifier for validation with the specified random walk strategy
    classifier = RandomWalkClassifier(valid_graph, random_walk_name, device=device)

    valid_dataloader = DataLoader(valid_dataset, batch_size=8)

    # Evaluate the model on validation data
    sys.stdout.write("Evaluating the model on validation data...\n")
    classifier.evaluate(valid_dataloader)
    sys.stdout.write("Validation evaluation completed.\n")

    # Initialize classifier for testing with the specified random walk strategy
    classifier = RandomWalkClassifier(test_graph, random_walk_name, device=device)

    # Generate test walks
    sys.stdout.write("Generating test walks...\n")
    test_valid_walks = classifier.generate_random_walks(num_walks=100, walk_length=5)
    test_invalid_walks = classifier.generate_invalid_random_walks(num_walks=100, walk_length=5)
    sys.stdout.write(
        f"Generated {len(test_valid_walks)} valid test walks and {len(test_invalid_walks)} invalid test walks.\n")
    sys.stdout.write(f"Valid test walks: {test_valid_walks}\n")
    sys.stdout.write(f"Invalid test walks: {test_invalid_walks}\n")

    test_dataset = classifier.prepare_data(test_valid_walks, test_invalid_walks)

    test_dataloader = DataLoader(test_dataset, batch_size=8)

    # Evaluate the model on test data
    sys.stdout.write("Evaluating the model on test data...\n")
    classifier.evaluate(test_dataloader)
    sys.stdout.write("Test evaluation completed.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Random Walk Classifier')
    parser.add_argument('--random_walk', type=str, required=True,
                        help='Name of the random walk strategy (BrownianMotion, ERGRW)')
    parser.add_argument('--tune', action='store_true', help='Flag to enable hyperparameter tuning')
    args = parser.parse_args()
    main(args.random_walk, args.tune)

    # Running script:
    # python -m random_walk_classifier.run_classifier --random_walk <name_of_random_walk_strategy> [--tune]
    # python -m random_walk_classifier.run_classifier --random_walk BrownianMotion --tune
    # python -m random_walk_classifier.run_classifier --random_walk ERGRW
