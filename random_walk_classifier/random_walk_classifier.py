import argparse
import os

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

    def predict(self, input_walks):
        self.model.eval()
        encoded_walks = [' '.join(map(str, walk)) for walk in input_walks]
        inputs = self.tokenizer(encoded_walks, return_tensors='pt', padding=True, truncation=True, max_length=512).to(
            self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()

        return predictions


def main(random_walk_name, tune, alpha, num_walks, walk_length, batch_size, parent_output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize Codex
    codex = Codex(code="en", size="s")

    # Initialize and load the Graph
    graph = Graph(codex).load_graph()

    # Initialize classifier
    classifier = RandomWalkClassifier(device=device)

    # Initialize RandomWalkGenerator
    generator = RandomWalkGenerator(graph)

    # Prepare datasets
    train_dataset, valid_dataset, test_dataset = (DataPreparation(generator, classifier, codex)
                                                  .prepare_datasets(random_walk_name, alpha, num_walks, walk_length))

    # Create output directory based on hyperparameters
    output_dir = os.path.join(parent_output_dir,
                              f"{random_walk_name.lower()}_alpha{alpha}_walks{num_walks}_length{walk_length}_batch{batch_size}")
    print(f"Creating output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Train and Evaluate Model
    model_trainer = ModelTrainer(output_dir, classifier, tune, train_dataset, valid_dataset, test_dataset, device, batch_size)
    model_trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Random Walk Classifier')
    parser.add_argument('--random_walk', type=str, default=None,
                        help='Name of the random walk strategy (Traditional, ERGRW, ERGRW_Adapted), optional')
    parser.add_argument('--tune', action='store_true', help='Flag to tune hyperparameters')
    parser.add_argument('--alpha', type=float, default=0.5, help='Alpha parameter for the ERGRW random walk generator')
    parser.add_argument('--num_walks', type=int, default=4000, help='Number of walks to generate')
    parser.add_argument('--walk_length', type=int, default=6, help='Length of each walk')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for data loading')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the results')
    args = parser.parse_args()

    main(args.random_walk, args.tune, args.alpha, args.num_walks, args.walk_length, args.batch_size, args.output_dir)
    # Running script:
    # python random_walk_classifier/random_walk_classifier.py
    #       --random_walk Traditional --tune --alpha 0.6 --num_walks 5000 --walk_length 8 --batch_size 16
    #       --parent_output_dir /content/drive/MyDrive/codex_random_walk

