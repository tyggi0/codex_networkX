import torch
from transformers import BertTokenizer, BertForSequenceClassification

from codex.codex import Codex
from graph import Graph
from random_walk_generator import RandomWalkGenerator
from data_preparation import DataPreparation
from model_trainer import ModelTrainer


class RandomWalkClassifier:
    def __init__(self, device):
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(self.device)


def main(random_walk_name, tune, num_walks=4000, walk_length=6):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize Codex
    codex = Codex(code="en", size="s")

    # Initialize and load the Graph
    graph = Graph(codex).load_graph()

    # Initialize classifier
    classifier = RandomWalkClassifier(device=device)

    # Initialize RandomWalkGenerator
    generator = RandomWalkGenerator(graph, random_walk_name)

    train_dataset, valid_dataset, test_dataset = DataPreparation(generator, classifier, codex, num_walks,
                                                                 walk_length).prepare_datasets()

    model_trainer = ModelTrainer(classifier, train_dataset, valid_dataset, test_dataset, device, batch_size=32,
                                 tune=tune).train()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Random Walk Classifier')
    parser.add_argument('--random_walk', type=str, required=True,
                        help='Name of the random walk strategy (BrownianMotion, ERGRW)')
    parser.add_argument('--tune', action='store_true', help='Flag to tune hyperparameters')
    args = parser.parse_args()
    main(args.random_walk, args.tune)

    # Running script:
    # python random_walk_classifier/random_walk_classifier.py --random_walk <name_of_random_walk_strategy> [--tune]
    # python random_walk_classifier/random_walk_classifier.py --random_walk BrownianMotion
    # python random_walk_classifier/random_walk_classifier.py --random_walk BrownianMotion --tune
    # python random_walk_classifier/random_walk_classifier.py --random_walk ERGRW
    # python random_walk_classifier/random_walk_classifier.py --random_walk ERGRW --tune
