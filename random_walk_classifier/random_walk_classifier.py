import argparse
import os
import random

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification

from codex.codex import Codex
from graph import Graph
from random_walk_generator import RandomWalkGenerator
from data_preparation import DataPreparation
from model_trainer import ModelTrainer


class RandomWalkClassifier(nn.Module):
    def __init__(self, device, dropout_rate=0.1):
        super(RandomWalkClassifier, self).__init__()
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(self.device)
        self.dropout = nn.Dropout(dropout_rate)  # Dropout layer
        self.classifier = nn.Linear(self.model.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output

        # Apply dropout
        dropped_output = self.dropout(pooled_output)

        # Final classification
        logits = self.classifier(dropped_output)
        return logits


def create_output_dir(random_walk_name, tune, alpha, num_walks, walk_length, description, lowercase,
                      encoding_format, size, dataset_mode, n_iterations, early_drop, parent_output_dir):
    tune_str = "tune_" if tune else ""
    alpha_str = f"_alpha{alpha}" if random_walk_name and random_walk_name != "traditional" else ""
    random_walk_name_str = f"{random_walk_name}" if random_walk_name else "codex"
    description_str = "_description" if description else ""
    lowercase_str = "_lowercase" if lowercase else ""
    size_str = f"_{size}size" if size and dataset_mode == "" else ""
    early_drop_str = "_early_drop" if early_drop else ""
    iterations_str = f"_{n_iterations}iterations" if tune else ""

    output_dir = os.path.join(
        parent_output_dir,
        f"{tune_str}{random_walk_name_str}{alpha_str}_walks{num_walks}_length{walk_length}"
        f"{description_str}{lowercase_str}_{encoding_format}{size_str}_{dataset_mode}"
        f"{iterations_str}{early_drop_str}")

    print(f"Creating output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def main(random_walk_name, tune, alpha, num_walks, walk_length, description, lowercase, encoding_format, size,
         dataset_mode, n_iterations, early_drop, parent_output_dir):
    random.seed(34)

    random_walk_name = random_walk_name.lower() if random_walk_name else ""
    encoding_format = encoding_format.lower() if encoding_format else ""
    size = size.lower() if size else ""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create output directory based on hyperparameters
    output_dir = create_output_dir(random_walk_name, tune, alpha, num_walks,
                                   walk_length, description, lowercase, encoding_format,
                                   size, dataset_mode, n_iterations, early_drop, parent_output_dir)

    # Initialize Codex
    codex = Codex(code="en", size="s")

    # Initialize and load the Graph
    graph = Graph(codex, description, lowercase).load_graph()

    # Initialize classifier
    classifier = RandomWalkClassifier(device=device)

    # Initialize RandomWalkGenerator
    generator = RandomWalkGenerator(graph)

    # Prepare datasets
    train_dataset, valid_dataset, test_dataset = (
        DataPreparation(generator, classifier, codex, description, encoding_format, lowercase)
        .prepare_datasets(random_walk_name, alpha, num_walks, walk_length, size, dataset_mode))

    # Train and Evaluate Model
    model_trainer = ModelTrainer(output_dir, classifier, tune,
                                 train_dataset, valid_dataset, test_dataset, n_iterations, early_drop, device)
    model_trainer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Random Walk Classifier')
    parser.add_argument('--random_walk', type=str, default=None,
                        help='Name of the random walk strategy (Traditional, ERGRW, Adapted_ERGRW), optional')
    parser.add_argument('--tune', action='store_true', help='Flag to tune hyperparameters')
    parser.add_argument('--alpha', type=float, default=0.5, help='Alpha parameter for the ERGRW random walk generator')
    parser.add_argument('--num_walks', type=int, default=8200, help='Number of valid walks to generate')
    parser.add_argument('--walk_length', type=int, default=6, help='Length of each walk')
    parser.add_argument('--description', action='store_true',
                        help='Flag to include description for data textual representation')
    parser.add_argument('--lowercase', action='store_true', help='Flag to convert train data to lowercase')
    parser.add_argument('--encoding_format', type=str, default="bert",
                        help='Name of the encoding format (BERT or Tag)')
    parser.add_argument('--size', type=str, default="full", help='CODEX train dataset size (full or half)')
    parser.add_argument('--dataset_mode', type=str, default="random_walks_only",
                        help='Dataset preparation mode (codex_only, random_walks_only, or combined)')
    parser.add_argument('--n_iterations', type=int, default=7,
                        help='Specify the number of iterations to perform during hyperparameter tuning. ')
    parser.add_argument('--early_drop', action='store_true',
                        help='Enable early stopping during training to halt training when validation performance does not improve after 3 epochs.')
    parser.add_argument('--parent_output_dir', type=str,
                        default="/content/drive/MyDrive/codex_random_walk", help='Directory to save the results')
    args = parser.parse_args()

    main(args.random_walk, args.tune, args.alpha, args.num_walks, args.walk_length, args.description,
         args.lowercase, args.encoding_format, args.size, args.dataset_mode, args.n_iterations, args.early_drop,
         args.parent_output_dir)

    # Running script:
    # python random_walk_classifier/random_walk_classifier.py
    #       --random_walk Traditional --tune --alpha 0.6 --num_walks 5000 --walk_length 8
    #       --description --lowercase --encoding_format bert --size half
    #       --parent_output_dir /content/drive/MyDrive/codex_random_walk
