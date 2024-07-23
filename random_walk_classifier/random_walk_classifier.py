import random
import torch
import networkx as nx
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from torch.utils.data import Dataset
from brownian_motion_random_walk import BrownianMotionRandomWalk
from ergrw_random_walk import ERGRWRandomWalk
from codex.codex import Codex
from sklearn.model_selection import ParameterGrid


class RandomWalkClassifier:
    def __init__(self, graph: nx.Graph, random_walk_strategy, device):
        self.graph = graph
        self.random_walk = self.get_random_walk_strategy(random_walk_strategy)
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(self.device)

    def get_random_walk_strategy(self, name):
        if name == "BrownianMotion":
            return BrownianMotionRandomWalk(self.graph)
        elif name == "ERGRW":
            return ERGRWRandomWalk(self.graph)
        else:
            raise ValueError(f"Unknown random walk strategy: {name}")

    def generate_random_walks(self, num_walks, walk_length):
        return self.random_walk.generate_walks(num_walks, walk_length)

    def generate_invalid_random_walks(self, num_walks, walk_length):
        nodes = list(self.graph.nodes)
        invalid_walks = []
        for _ in range(num_walks):
            walk = []
            current_node = random.choice(nodes)
            for _ in range(walk_length):
                walk.append(current_node)
                if random.random() > 0.5:
                    current_node = random.choice(nodes)
                else:
                    neighbors = list(self.graph.neighbors(current_node))
                    if neighbors:
                        current_node = random.choice(neighbors)
                    else:
                        break
            invalid_walks.append(walk)
        return invalid_walks

    def encode_walks(self, walks):
        return [self.tokenizer.encode(' '.join(map(str, walk)), add_special_tokens=True) for walk in walks]

    def prepare_data(self, valid_walks, invalid_walks):
        encoded_valid_walks = self.encode_walks(valid_walks)
        encoded_invalid_walks = self.encode_walks(invalid_walks)

        labels_valid = [1] * len(encoded_valid_walks)
        labels_invalid = [0] * len(encoded_invalid_walks)

        walks = encoded_valid_walks + encoded_invalid_walks
        labels = labels_valid + labels_invalid

        return walks, labels

    class WalkDataset(Dataset):
        def __init__(self, walks, labels, tokenizer):
            self.walks = walks
            self.labels = labels
            self.tokenizer = tokenizer

        def __len__(self):
            return len(self.walks)

        def __getitem__(self, idx):
            walk = self.walks[idx]
            label = self.labels[idx]
            encoding = self.tokenizer(walk, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
            return {**encoding, 'labels': torch.tensor(label, dtype=torch.long)}


def construct_labeled_graph(triples, codex):
    G = nx.Graph()
    for head, relation, tail in triples.values:
        head_label = codex.entity_label(head)
        relation_label = codex.relation_label(relation)
        tail_label = codex.entity_label(tail)
        G.add_edge(head_label, tail_label, relation=relation_label)
    return G


def prepare_datasets(classifier, graph, num_walks, walk_length):
    valid_walks = classifier.generate_random_walks(num_walks, walk_length)
    invalid_walks = classifier.generate_invalid_random_walks(num_walks, walk_length)
    walks, labels = classifier.prepare_data(valid_walks, invalid_walks)
    return RandomWalkClassifier.WalkDataset(walks, labels, classifier.tokenizer)


def tune_hyperparameters(trainer, train_dataset, valid_dataset):
    param_grid = {
        'learning_rate': [1e-5, 3e-5, 5e-5],
        'num_train_epochs': [2, 3, 4],
        'per_device_train_batch_size': [16, 32]
    }
    best_params = None
    best_score = float('-inf')

    for params in ParameterGrid(param_grid):
        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            learning_rate=params['learning_rate'],
            per_device_train_batch_size=params['per_device_train_batch_size'],
            per_device_eval_batch_size=32,
            num_train_epochs=params['num_train_epochs'],
            weight_decay=0.01,
            logging_dir='./logs',
        )

        trainer.args = training_args
        trainer.train_dataset = train_dataset
        trainer.eval_dataset = valid_dataset

        trainer.train()
        eval_results = trainer.evaluate()

        if eval_results['eval_accuracy'] > best_score:
            best_score = eval_results['eval_accuracy']
            best_params = params

    print(f"Best parameters found: {best_params} with accuracy {best_score}")
    return best_params


def main(random_walk_name, tune):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize Codex
    codex = Codex(code="en", size="s")

    # Access entities, relations, and triples
    entities = codex.entities()
    relations = codex.relations()
    triples = codex.triples()

    print(f"Number of entities: {len(entities)}")
    print(f"Number of relations: {len(relations)}")
    print(f"Number of triples: {len(triples)}")

    # Access train, validation, and test triples
    train_triples = codex.split("train")
    valid_triples = codex.split("valid")
    test_triples = codex.split("test")

    print("Constructing labeled graphs...")
    # Construct graphs with labeled triples
    train_graph = construct_labeled_graph(train_triples, codex)
    valid_graph = construct_labeled_graph(valid_triples, codex)
    test_graph = construct_labeled_graph(test_triples, codex)
    print("Graphs constructed successfully.")

    # Initialize classifier with the specified random walk strategy
    classifier = RandomWalkClassifier(train_graph, random_walk_name, device=device)

    # Prepare datasets
    print("Preparing train dataset...")
    train_dataset = prepare_datasets(classifier, train_graph, num_walks=500, walk_length=5)
    print("Preparing validation dataset...")
    valid_dataset = prepare_datasets(classifier, valid_graph, num_walks=100, walk_length=5)
    print("Preparing test dataset...")
    test_dataset = prepare_datasets(classifier, test_graph, num_walks=100, walk_length=5)

    # Set up the training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=classifier.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=classifier.tokenizer,
    )

    if tune:
        print("Tuning hyperparameters...")
        best_params = tune_hyperparameters(trainer, train_dataset, valid_dataset)
        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            learning_rate=best_params['learning_rate'],
            per_device_train_batch_size=best_params['per_device_train_batch_size'],
            per_device_eval_batch_size=32,
            num_train_epochs=best_params['num_train_epochs'],
            weight_decay=0.01,
        )
        trainer.args = training_args

    # Train the model
    print("Training the model...")
    trainer.train()
    print("Training completed.")

    # Evaluate the model on test data
    print("Evaluating the model on test dataset...")
    trainer.evaluate(eval_dataset=test_dataset)
    print("Test evaluation completed.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Random Walk Classifier')
    parser.add_argument('--random_walk', type=str, required=True, help='Name of the random walk strategy (BrownianMotion, ERGRW)')
    parser.add_argument('--tune', action='store_true', help='Flag to tune hyperparameters')
    args = parser.parse_args()
    main(args.random_walk, args.tune)

    # Running script:
    # python random_walk_classifier/random_walk_classifier.py --random_walk <name_of_random_walk_strategy> [--tune]
    # python random_walk_classifier/random_walk_classifier.py --random_walk BrownianMotion
    # python random_walk_classifier/random_walk_classifier.py --random_walk BrownianMotion --tune
    # python random_walk_classifier/random_walk_classifier.py --random_walk ERGRW
    # python random_walk_classifier/random_walk_classifier.py --random_walk ERGRW --tune
