import networkx as nx
from codex.codex import Codex
import torch
from torch.utils.data import DataLoader
from random_walk_classifier.random_walk_classifier import RandomWalkClassifier
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def construct_labeled_graph(triples, codex):
    G = nx.Graph()
    for head, relation, tail in triples.values:
        head_label = codex.entity_label(head)
        relation_label = codex.relation_label(relation)
        tail_label = codex.entity_label(tail)
        G.add_edge(head_label, tail_label, relation=relation_label)
    return G


def main(random_walk_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize Codex
    codex = Codex(code="en", size="s")

    # Access entities, relations, and triples
    entities = codex.entities()
    relations = codex.relations()
    triples = codex.triples()

    logger.info("Number of entities: %d", len(entities))
    logger.info("Number of relations: %d", len(relations))
    logger.info("Number of triples: %d", len(triples))

    # Access train, validation, and test triples
    train_triples = codex.split("train")
    valid_triples = codex.split("valid")
    test_triples = codex.split("test")

    logger.info("Constructing labeled graphs...")
    # Construct graphs with labeled triples
    train_graph = construct_labeled_graph(train_triples, codex)
    valid_graph = construct_labeled_graph(valid_triples, codex)
    test_graph = construct_labeled_graph(test_triples, codex)
    logger.info("Graphs constructed successfully.")

    # Initialize classifier with the specified random walk strategy
    classifier = RandomWalkClassifier(train_graph, random_walk_name, device=device)

    # Generate train walks
    logger.info("Generating train walks...")
    train_valid_walks = classifier.generate_random_walks(num_walks=500, walk_length=5)
    train_invalid_walks = classifier.generate_invalid_random_walks(num_walks=500, walk_length=5)
    logger.info(
        f"Generated {len(train_valid_walks)} valid train walks and {len(train_invalid_walks)} invalid train walks.")
    logger.info(f"Valid train walks: {train_valid_walks}")
    logger.info(f"Invalid train walks: {train_invalid_walks}")

    train_walks, train_labels = classifier.prepare_data(train_valid_walks, train_invalid_walks)
    logger.info(f"Sample encoded train walk: {train_walks[0]}, Label: {train_labels[0]}")

    train_dataset = RandomWalkClassifier.WalkDataset(train_walks, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    # Train the model
    logger.info("Training the model...")
    classifier.train(train_dataloader)
    logger.info("Training completed.")

    # Initialize classifier for validation with the specified random walk strategy
    classifier = RandomWalkClassifier(valid_graph, random_walk_name, device=device)

    # Generate validation walks
    logger.info("Generating validation walks...")
    valid_valid_walks = classifier.generate_random_walks(num_walks=100, walk_length=5)
    valid_invalid_walks = classifier.generate_invalid_random_walks(num_walks=100, walk_length=5)
    logger.info(
        f"Generated {len(valid_valid_walks)} valid validation walks and {len(valid_invalid_walks)} invalid validation walks.")
    logger.info(f"Valid validation walks: {valid_valid_walks}")
    logger.info(f"Invalid validation walks: {valid_invalid_walks}")

    valid_walks, valid_labels = classifier.prepare_data(valid_valid_walks, valid_invalid_walks)
    logger.info(f"Sample encoded validation walk: {valid_walks[0]}, Label: {valid_labels[0]}")

    valid_dataset = RandomWalkClassifier.WalkDataset(valid_walks, valid_labels)
    valid_dataloader = DataLoader(valid_dataset, batch_size=8, shuffle=False)

    # Evaluate the model on validation data
    logger.info("Evaluating the model on validation data...")
    classifier.evaluate(valid_dataloader)
    logger.info("Validation evaluation completed.")

    # Initialize classifier for testing with the specified random walk strategy
    classifier = RandomWalkClassifier(test_graph, random_walk_name, device=device)

    # Generate test walks
    logger.info("Generating test walks...")
    test_valid_walks = classifier.generate_random_walks(num_walks=100, walk_length=5)
    test_invalid_walks = classifier.generate_invalid_random_walks(num_walks=100, walk_length=5)
    logger.info(f"Generated {len(test_valid_walks)} valid test walks and {len(test_invalid_walks)} invalid test walks.")
    logger.info(f"Valid test walks: {test_valid_walks}")
    logger.info(f"Invalid test walks: {test_invalid_walks}")

    test_walks, test_labels = classifier.prepare_data(test_valid_walks, test_invalid_walks)
    logger.info(f"Sample encoded test walk: {test_walks[0]}, Label: {test_labels[0]}")

    test_dataset = RandomWalkClassifier.WalkDataset(test_walks, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Evaluate the model on test data
    logger.info("Evaluating the model on test data...")
    classifier.evaluate(test_dataloader)
    logger.info("Test evaluation completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Random Walk Classifier')
    parser.add_argument('--random_walk', type=str, required=True,
                        help='Name of the random walk strategy (BrownianMotion, ERGRW)')
    args = parser.parse_args()
    main(args.random_walk)

    # Running script:
    # python random_walk_classifier/random_walk_classifier.py --random_walk <name_of_random_walk_strategy>
    # python random_walk_classifier/random_walk_classifier.py --random_walk BrownianMotion
    # python random_walk_classifier/random_walk_classifier.py --random_walk ERGRW